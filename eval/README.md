# Evaluation

- `corpus/sample/`: four short, made-up policy documents for **smoke runs** (checking the pipeline and harness end to end). They are too small to measure quality; the real evaluation corpus (50–200 public documents) is chosen in Phase 0 and the full labelled dataset is built in Phase 9.
- `datasets/sample_qa.jsonl`: 14 labelled questions for the sample corpus, covering exact terms, paraphrases, single facts, a numeric question, a multi-hop question and two unanswerable ones. Each gold reference is a file, a page and optionally a section heading.
- `reports/`: generated results (Markdown + JSON), named `<timestamp>-<git sha>-retrieval`. Each report compares `keyword`, `vector`, `hybrid` and `hybrid+rerank`, shows latency (including rerank time) and includes a table for choosing `RERANK_MIN_SCORE`.

## Run a retrieval evaluation locally

```bash
docker compose up -d --build
cd backend
python -m app.evaluation.load_corpus ../eval/corpus/sample --api http://localhost:8000
docker compose exec api python -m app.evaluation.retrieval \
  --dataset /eval/datasets/sample_qa.jsonl --out /eval/reports --k 1 3 5 10
```

The evaluation runs inside the `api` container, so it uses the same model server and settings as the app; `./eval` is mounted at `/eval`. Add `--configs hybrid hybrid+rerank` to run only some configurations. It searches as the dev user who uploaded the corpus.
