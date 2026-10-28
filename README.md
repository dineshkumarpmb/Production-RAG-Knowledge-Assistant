# Production-RAG-Knowledge-Assistant
Businesses need to search large collections of PDFs, policies, reports, and technical documentation and get answers grounded in their content.

See [PLAN.md](PLAN.md) for the full implementation plan. **Everything runs locally** with Docker Compose; there is no cloud deployment.

## Status

| Phase | State |
|-------|-------|
| 0 Foundations | Partial: backend skeleton, Docker Compose, Alembic. CI, frontend and eval corpus still to do |
| 1 Ingestion and metadata | Done |
| 2 Chunking, embeddings, duplicate detection | Done |
| 3 Hybrid retrieval | Done (first eval run on local bge-m3: see results below) |
| 4 Reranking and context assembly | Done |
| 5 Grounded answers with citations, streaming | Done (live Claude calls untested: no API key on this machine) |
| 6 Authentication, permissions, security | Done |
| 7 Conversations, follow-ups, error handling | Done |
| 8 to 12 | Not started (11 is local packaging, not cloud deployment) |

## Quick start

```bash
# Local models by default (bge-m3 + a cross-encoder); the first start downloads about 2.5 GB.
# Hosted instead: EMBEDDING_PROVIDER=voyage RERANKER_PROVIDER=voyage VOYAGE_API_KEY=...
docker compose up --build
```

The API runs at http://localhost:8000 (interactive docs at `/docs`). Postgres is published on host port **5433** (set `POSTGRES_PORT` to change it), so it does not clash with a Postgres already installed on the machine. Then:

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/documents   # 202 + document id
curl http://localhost:8000/documents/<id>                       # status: queued -> parsing -> embedding -> ready | failed
curl http://localhost:8000/documents/<id>/pages                 # cleaned text per page, with heading positions
curl http://localhost:8000/documents/<id>/chunks                # chunks with page ranges and heading paths
curl -X POST http://localhost:8000/documents/<id>/reprocess     # re-run ingestion (idempotent)
curl -X DELETE http://localhost:8000/documents/<id>             # delete a document, its chunks and file
curl http://localhost:8000/admin/ingestion-report               # chunk counts, duplicates, embedding cost
curl -H "Content-Type: application/json" -d '{"query": "hotel limit in Paris"}'      http://localhost:8000/search                                # hybrid search with page ranges
curl -H "Content-Type: application/json" -d '{"query": "SEC-4471"}'      http://localhost:8000/admin/search/debug                    # keyword, vector and fused rankings
```

Every endpoint except `/healthz`, `/readyz` and `/auth/*` needs a bearer token. Create an organisation with `POST /auth/register` (its first user becomes the admin), then sign in with `POST /auth/login` (form fields `username` = email, `password`) and send `Authorization: Bearer <access_token>`. The interactive docs at `/docs` have an Authorize button for this flow.

## Development

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate   # or: source .venv/bin/activate
pip install -e ".[dev]"
pytest              # SQLite and an in-process queue; Postgres-only tests are skipped
ruff check . && ruff format --check . && mypy app

# Full suite on real Postgres + pgvector (includes search tests). The database is wiped.
docker compose up -d postgres
docker compose exec postgres psql -U rag -c "CREATE DATABASE rag_test"   # once
TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5433/rag_test pytest
```

## Ingestion (Phase 1)

- **Upload checks:** supported extension (`.pdf`, `.docx`, `.txt`, `.md`) **and** matching content (PDF header, a real DOCX package without macros or zip bombs, no binary data in text files); size limit; safe file names; SHA-256 hash. Re-uploading the exact same file returns the existing document instead of storing a copy.
- **Parsing:** PyMuPDF for PDFs (reading order across two columns, headings from the PDF outline or font sizes, title, author and dates); python-docx for DOCX (heading styles, tables); Markdown headings for `.md`.
- **DOCX page numbers:** Word saves its last page layout as `lastRenderedPageBreak` markers, and pages are split on those when present (`page_mode: rendered`). Files without them, such as generated ones, are split at explicit page breaks and about every 3,000 characters (`page_mode: virtual`).
- **Clean-up:** Unicode NFC with ligature expansion (not NFKC, which would turn `x²` into `x2`); words split by a hyphen at a line end are rejoined; running headers and footers in the page margins are removed; whitespace is collapsed.
- **Flags:** language detection; pages with almost no text are listed in `metadata.low_text_pages`, and `metadata.needs_ocr` is set for mostly-empty PDFs. Files with no text at all fail with a clear reason.
- **Reliability:** Celery worker with late acknowledgement and retries for database errors; re-running ingestion replaces a document's pages in one transaction.

## Chunking, embeddings and duplicates (Phase 2)

- **Chunking:** headings start new sections and chunks never cross them; paragraphs are packed into chunks of about 500 tokens with 15% overlap; paragraphs that are too long are split by line, then sentence, then word. Each chunk keeps its page range, character offsets and heading path (for example `Handbook > Leave > Sick leave`). Token counts are approximate (words and punctuation), which is accurate enough for sizing.
- **Contextual header:** the text sent for embedding is prefixed with `title > heading path`; the stored chunk text is not. This can be switched off with `CHUNK_CONTEXTUAL_HEADER=false` for the Phase 9 comparison.
- **Embeddings:** one interface with two providers: Voyage AI (`voyage-4`, the default, $0.06 per million tokens) and local `BAAI/bge-m3`. Both produce 1024-dimension vectors, stored in pgvector with an HNSW index (`m=16, ef_construction=64`). Requests are batched, rate-limited and retried with exponential backoff and jitter; `Retry-After` is respected. Client errors such as 400 or 401 are not retried.
- **Embedding cache:** keyed by `(SHA-256 of the exact text sent, model)`, so reprocessing a document or uploading repeated text costs nothing extra. Each batch is cached as soon as it returns, so a failure partway through keeps what was already paid for.
- **Changing the model:** every chunk records the model that embedded it. After changing `EMBEDDING_MODEL`, run `celery -A app.workers.celery_app:celery_app call ingestion.reembed_stale` to re-embed affected documents without parsing them again.
- **Duplicates:**
  - Exact: same SHA-256 within a tenant. The existing document is returned and a duplicate-upload counter goes up.
  - Near-duplicate: the 64-bit SimHash of the full text differs by 3 bits or fewer. The new document is processed as normal and flagged through `near_duplicate_of_id`. The owner then calls `POST /documents/{id}/near-duplicate` with `keep_both` or `replace`; `replace` deletes the older document and increments the version. Only documents the uploader can see are compared.
  - Chunk: the hash of the normalised chunk text, counted in the ingestion report so repeated boilerplate can be collapsed in Phase 3.
- **Ingestion report:** `GET /admin/ingestion-report` shows documents by status, pages, chunks, average chunk size, duplicate chunks, exact and near-duplicate counts, and embedding tokens, cost and cache hits.

## Hybrid retrieval (Phase 3)

- **Keyword search:** Postgres full-text search on a generated `tsvector` column (heading path weighted above body text) with a GIN index, ranked by `ts_rank_cd`. Plain queries match **any** word, so chunks matching more words rank higher; requiring every word would miss most chunks for question-style input. Queries using search syntax (`"exact phrase"`, `-exclude`, upper-case `OR`) are applied strictly.
- **Vector search:** cosine distance on the HNSW index, comparing only chunks embedded with the same model as the query. `hnsw.ef_search` is configurable, and **iterative index scans** are on, so documents the user cannot read never crowd out ones they can. A test confirms this: with iterative scans switched off, it returns nothing.
- **Access control in the query:** keyword search, vector search and the documents API all use one rule (`app/db/access.py`), applied in SQL before any ranking. Tests check that another user in the same tenant, and any user in another tenant, get no results from private documents in any mode.
- **Fusion:** Reciprocal Rank Fusion (`k = 60`) with optional per-retriever weights. Keyword search and query embedding plus vector search run at the same time. Chunks with identical text (boilerplate) are collapsed into one result with a `duplicates` count.
- **Degraded mode:** if the embedding provider is down, hybrid search returns keyword results and reports `degraded: ["vector_unavailable"]`; vector-only search returns 503.
- **Endpoints:** `POST /search` for any user (filters: document IDs, file types, date range; modes `hybrid`, `keyword`, `vector`), and `POST /admin/search/debug` showing the three rankings side by side with adjustable RRF `k`, weights, candidate count and `ef_search`.
- **Evaluation harness:** `python -m app.evaluation.retrieval` reports Recall@1/5/10/20, Hit@k, MRR, nDCG@10 and p50/p95 latency per mode, overall and by question type, and writes Markdown and JSON to `eval/reports/`. See [eval/README.md](eval/README.md) for the sample corpus and how to run it.

## Reranking and context assembly (Phase 4)

- **Reranker interface** with a local cross-encoder (sentence-transformers) and Voyage `rerank-2.5` (optional, hosted). Search fuses the top 30 candidates (`RERANK_CANDIDATES`), the cross-encoder rescores them by reading query and chunk together, and the best `RERANK_TOP_K` (8) are kept. The reranker sees each chunk with its `title > heading path` prefix.
- **"Not found" threshold:** if the best reranked score is below `RERANK_MIN_SCORE`, retrieval reports `not_found` and returns no context, so the answer step can say "not found in your documents" without calling the LLM. It is off by default; the evaluation report includes a table to pick the value from (share of unanswerable questions correctly refused vs answerable ones wrongly refused at each threshold).
- **Context assembly:** chunks that are next to each other in a document are merged into one passage, with the overlap between them removed. Passages are ordered by relevance and kept within `CONTEXT_TOKEN_BUDGET` (3000 tokens), and each keeps its chunk IDs, pages and heading path for citations.
- **Degraded mode:** if the reranker is unavailable, results keep the fused order and report `degraded: ["rerank_unavailable"]`.
- **Endpoints:** `POST /search` with `"rerank": true` shows each hit's `rerank_score` and its `fused_rank` before reranking. `POST /retrieve` returns the assembled passages that Phase 5 will send to the model. The admin debug endpoint accepts `rerank`, `rerank_candidates` and `rerank_min_score`.
- **Model server:** the `models` container loads the embedding model and the reranker once, and the API and worker call it over HTTP (`MODEL_SERVER_URL`). Loading the models inside both the API and the worker would double the memory. Models are cached in the `hf_cache` volume after the first download.
- **Cost and latency:** rerank time is reported separately (`timings_ms.rerank`, and p50 in the eval report), and the cost of a hosted reranker is reported as `rerank_cost_usd`.
- **Default local models:** `BAAI/bge-m3` embeddings and `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters, English). `BAAI/bge-reranker-v2-m3` is stronger and multilingual, but it is as large as bge-m3 (about 2 GB) and does not fit next to it on a 6 GB laptop. Set `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` on a machine with more memory.

### First results with real local models (sample set)

From `eval/reports/20261028-155647-nogit-retrieval.md`: 4 documents, 13 chunks, 12 answerable and 2 unanswerable questions, run on a 4-core laptop (Ryzen 5 3550H, 6 GB RAM, about 2.8 GB available to Docker).

| Config | Recall@1 | Recall@3 | MRR | nDCG@10 | p50 latency |
|---|---:|---:|---:|---:|---:|
| keyword | 0.750 | 1.000 | 0.875 | 0.913 | 9 ms |
| vector (bge-m3) | 0.958 | 1.000 | 1.000 | 1.000 | 8.4 s |
| hybrid | 0.958 | 1.000 | 1.000 | 1.000 | 6.9 s |
| hybrid + rerank (MiniLM) | 0.875 | 1.000 | 0.958 | 0.969 | 1.3 s (rerank 0.5 s) |

What this does and does not show:
- **The sample is too small to rank configurations.** Recall@3 is 1.0 everywhere, and one question equals 0.083 of Recall@1. Real comparisons need the Phase 9 dataset.
- **Vector search fixes keyword's paraphrase misses** (q05, q07, q10 move from rank 2 to rank 1).
- **The small reranker did not help here.** It dropped q04 ("How much vacation can I take each year?") to rank 2. It also gave correct paraphrase answers very low scores (0.002–0.07, while exact-wording matches score about 0.999), so `RERANK_MIN_SCORE` should stay off with this model: at 0.05 it would refuse both unanswerable questions but also 25% of answerable ones.
- **Latency is limited by memory, not the code.** With bge-m3 at 1.8 GB and only about 0.5 GB of RAM free on the host, query embedding took 1–9 s depending on swapping (p95 15 s). Keyword search takes milliseconds. A machine with 16 GB of RAM, or the hosted Voyage option, would be far faster.

## Grounded answers with citations (Phase 5)

```bash
CID=$(curl -s -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/chat | jq -r .id)
curl -N -H "Content-Type: application/json" -d '{"content": "What is the hotel limit in Paris?"}' \
     http://localhost:8000/chat/$CID/messages          # Server-Sent Events
curl http://localhost:8000/chat/$CID                    # stored messages with citations
```

- **Model:** `claude-opus-5-5` through the official `anthropic` SDK (set `ANTHROPIC_API_KEY`). Effort is set explicitly (`LLM_EFFORT=medium`, the model's default). Thinking is always on for this model, so no thinking settings are sent; sampling parameters (`temperature` and so on) are not sent because the API rejects them.
- **Citations:** each context passage is sent as a citable custom-content `document`, with each paragraph as its own content block. Claude's `content_block_location` citations map back to the exact paragraph, and so to its chunk IDs, document and pages. Citations are attached by the API rather than by asking the model to write `[1]` markers.
- **Grounding and prompt-injection defence:** a fixed system prompt (about 640 tokens, above the 512-token caching minimum, and cached with `cache_control`) tells Claude to answer only from the documents and to say "I couldn't find this in your documents" otherwise. It also says to treat document text as data and never follow instructions found in it. The documents sit in the user turn, never in the system prompt.
- **No answer without evidence:** if retrieval finds nothing, or the reranker threshold rules everything out, the reply says so and Claude is not called.
- **Streaming events:**
  - `sources`: the passages, sent before generation starts
  - `delta`: answer text
  - `citation`: a cited passage, with the offset where its span starts
  - `reset`: discard what was streamed so far; sent when a refusal or a fallback model restarts the answer
  - `done`: final text, cited spans, citations, citation coverage, model, stop reason, token usage, cost and timings
  - `error`: a typed code (`llm_not_configured`, `llm_rate_limited`, `llm_unavailable`, `llm_bad_request`, `internal_error`) and whether partial text was already sent
- **Refusals:** server-side `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) re-runs a declined request on the model Anthropic recommends for that refusal category. A final `stop_reason: "refusal"` replaces any partial output with a short refusal message. `max_tokens` marks the answer as truncated. Turn fallbacks off with `LLM_FALLBACKS=false`.
- **Cost:** computed from `usage`, including cache reads and writes. When a fallback ran, each attempt in `usage.iterations` is priced at its own model's rates. Prices live in `app/generation/pricing.py` (Opus 5.5: $4 / $20 per million input / output tokens, $0.20 for cache reads) and can be overridden with `LLM_PRICES`.
- **Stored:** the user and assistant messages (the assistant message with its citations, cited spans and metadata), and a `query_logs` row per question with retrieved and context chunk IDs, per-stage latency, tokens, cost, model, stop reason and any error.
- **Tests:** the SDK stream is replayed with fake events (cited answer, mid-stream fallback, refusal, truncation, each error class), the SSE contract is tested through the API, and chat runs end to end on Postgres with real retrieval. `tests/live/` calls the real API: a cited answer, prompt-cache reuse, "not found" for an unrelated question, and an injected instruction inside a document being ignored. These tests are skipped unless `ANTHROPIC_API_KEY` is set.

## Authentication, permissions and security (Phase 6)

- **Accounts:** `POST /auth/register` creates an organisation and its admin (disable with `ALLOW_REGISTRATION=false`). Admins create users (`viewer`, `editor`, `admin`) and groups under `/admin`. Passwords are hashed with argon2id, and a login attempt takes the same time whether or not the email exists.
- **Tokens:** access tokens are HS256 JWTs valid for 15 minutes, with the algorithm pinned and the token type checked. Refresh tokens are random, stored only as SHA-256 hashes and rotated on every use; replaying a rotated token revokes the whole session family. Role and active status are read from the database on every request, so demoting or deactivating a user takes effect immediately and also revokes their refresh tokens. The server refuses to start with `ENV=prod` and the development `JWT_SECRET`.
- **Roles:** viewers read and ask questions; editors also upload; admins also manage users, groups and the audit log. Admins get **no** extra read access to documents.
- **Sharing:** documents start private to the uploader. The owner (or an admin) can share with a user or group (`read` or `write`) via `POST /documents/{id}/access`, or with the whole organisation via `PATCH /documents/{id}` `{"visibility": "tenant"}`. `write` allows reprocessing and resolving duplicates; deleting and sharing stay with the owner. Anything a user cannot read answers **404**, so document IDs reveal nothing.
- **Two independent layers enforce reads:**
  1. Every query applies one rule (`app/db/access.py`): search, retrieval, chat, document pages and chunks, and file download.
  2. **Postgres row-level security** on `documents`, `document_pages` and `chunks`. The app connects as `rag_app` (not a superuser, no `BYPASSRLS`; superusers always bypass RLS), and each database session carries the user's scope in transaction-local settings. A session without a scope sees no rows. The ingestion worker uses an explicit `system` scope; admin reports and duplicate checks use a `tenant` scope.
- **Chat history:** citations and sources pointing at documents the user can no longer read (unshared or deleted) are replaced by a placeholder when history is loaded.
- **Web hardening:** per-user rate limits in Redis (chat 20/min, search 60/min, uploads 30/min, sign-in 10/min per email and IP; returns 429 with `Retry-After`). CORS is limited to `CORS_ORIGINS`. Security headers are set on every response (`nosniff`, `DENY` framing, no-referrer, a `default-src 'none'` CSP and `no-store` on API responses).
- **Audit log** (`GET /admin/audit`): sign-ins and failed sign-ins, refresh-token reuse, registration, uploads, downloads, sharing, visibility changes, deletions, replacements, user and group changes, and chat questions, each with user, target and client IP.
- **Upload safety** (from Phase 1): extension plus magic-byte checks, size limits, rejection of DOCX files with macros and of zip bombs, and safe file names. Uploaded content is never executed. ClamAV scanning is not included.

### Permissions matrix (zero leaks)

`tests/postgres/test_permissions_matrix.py` uploads one document per sharing mode, then checks six users across **ten channels**: document list, single document, pages, chunks, file download, keyword, vector and hybrid search, `/retrieve`, and chat sources. It runs through the API on Postgres with row-level security active. Every user must see exactly the ticked documents on every channel; an extra one fails the test as a leak, and a missing one fails it as broken sharing.

| User | private | group | user | tenant | other-org |
|---|:---:|:---:|:---:|:---:|:---:|
| alice (owner) | ✅ | ✅ | ✅ | ✅ | — |
| bob (Finance) | — | ✅ | — | ✅ | — |
| carol (shared) | — | — | ✅ | ✅ | — |
| dave (colleague) | — | — | — | ✅ | — |
| admin (Acme) | — | — | — | ✅ | — |
| eve (other org) | — | — | — | — | ✅ |

The documents are: *private* (Alice only), *group* (shared with Finance), *user* (shared with Carol), *tenant* (visible to everyone at Acme) and *other-org* (another organisation's tenant-wide document).

The other security tests:
- `tests/postgres/test_rls.py`: raw `SELECT`s with no application filter show each user exactly their rows; no scope means no rows; writes into another tenant are rejected; scopes don't leak between pooled connections.
- `tests/integration/test_sharing.py`: every document route answers 404 to outsiders, including another organisation's admin and the own organisation's admin; admin APIs are confined to their own organisation.
- `tests/integration/test_auth.py`: forged, expired, unsigned and wrong-type tokens; refresh reuse; rate limits; immediate role changes.
- `tests/unit/test_generation.py`: text from a poisoned document stays inside a document block and never reaches the system prompt. The live Claude tests check that the model ignores such text.

## Conversations, follow-up questions and error handling (Phase 7)

- **Follow-up questions:** before retrieval, a cheaper model (`claude-haiku-5-5`, set with `LLM_HELPER_MODEL`) rewrites the latest question into a standalone search query using the recent turns. For example, "what about contractors?" after a question on hotel limits becomes "hotel limit for contractors". The first question in a chat is used as is, at no cost. The rewritten query is sent in the `sources` event and stored in `query_logs.rewritten_query` next to the original. If rewriting fails, retrieval uses the question as asked and the response reports `degraded: ["rewrite_unavailable"]`.
- **What the answer model sees:** the user's original question, the most recent messages as plain text (`CHAT_HISTORY_MESSAGES`, default 6, each capped at `CHAT_HISTORY_MAX_CHARS`), and a running summary of older turns. The system prompt tells it to use earlier turns only to understand the question and to take facts only from the passages attached now.
- **Running summary:** once more than `CHAT_SUMMARY_AFTER_MESSAGES` (10) messages are unsummarised, the older ones are folded into `conversations.summary` by the helper model. This happens after the answer has been sent, so it adds no latency, and its cost is added to that turn's query log.
- **Conversations:** create, list, get, rename (`PATCH /chat/{id}`) and delete (`DELETE /chat/{id}`). Query logs of deleted chats are kept but unlinked, for cost and quality reporting.
- **Error handling:**

| Failure | Behaviour |
|---|---|
| Claude rate limit, server error or timeout | The SDK retries twice, then the stream ends with a typed `error` event; partial text is kept and marked |
| Embedding model or reranker down | Keyword-only search or no reranking; `degraded` says which |
| Query rewriting fails | The question is searched as asked; `degraded: ["rewrite_unavailable"]` |
| Nothing relevant found | "Not found in your documents", with no LLM call |
| Refusal | Server-side fallback model, otherwise a short refusal message |
| Ingestion fails | Document `failed` with the reason; `POST /documents/{id}/reprocess` retries |
| Client disconnects mid-answer | The upstream Claude stream is cancelled (no more tokens billed); the partial answer and the tokens used so far are still saved, marked `cancelled` |
| Unexpected bug while streaming | An `internal_error` event, and the partial answer is still saved |

- **Request IDs:** every response has `X-Request-ID`. The same ID is in the logs, in the `done` and `error` events, and in `query_logs.request_id`, so one failed answer can be traced end to end.
