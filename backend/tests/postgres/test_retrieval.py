"""Hybrid retrieval against real PostgreSQL + pgvector (skipped without TEST_DATABASE_URL)."""

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.access import Principal
from app.db.models import Chunk, Document, Role, Tenant, User, Visibility
from app.retrieval.queries import uses_search_syntax, vector_search
from app.retrieval.types import SearchFilters
from tests.conftest import FakeEmbeddingProvider, Identity, requires_postgres

pytestmark = requires_postgres

# Words that mean the same thing share a dimension, so paraphrases land close together.
CONCEPTS = [
    {"vacation", "holiday", "holidays", "pto", "leave"},
    {"password", "passwords", "credential", "credentials", "passphrase"},
    {"hotel", "hotels", "accommodation", "lodging"},
    {"laptop", "computer", "device", "stolen", "lost"},
]


class ConceptEmbedder(FakeEmbeddingProvider):
    """A tiny 'semantic' embedder: synonyms map to the same dimension."""

    def __init__(self) -> None:
        super().__init__(model_name="concept-embed-1")

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            for i, group in enumerate(CONCEPTS):
                if word in group:
                    v[i] += 1.0
            # Every word also nudges a hashed dimension, so unrelated texts still differ.
            v[10 + int(hashlib.md5(word.encode()).hexdigest(), 16) % (self.dim - 10)] += 0.05  # noqa: S324
        if not v.any():
            v[-1] = 1.0
        return [float(x) for x in v / np.linalg.norm(v)]


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return ConceptEmbedder()


LEAVE = """# Leave Policy

## Annual leave

Employees receive 25 days of paid vacation each year. Unused holiday can be carried over.

## Sick leave

Tell your manager before 9am on the first day of illness.
"""

SECURITY = """# Security Standard

## Incident codes

Incident code SEC-4471 means a lost or stolen laptop. Report it to the service desk at once.

## Credentials

Passwords must be at least 14 characters long and must never be shared.
"""

LAPTOP_DOC = "# Laptops\n\nReturn your laptop on leaving.\n"

EXPENSES = """# Expense Policy

## Travel

Hotels are reimbursed up to 150 EUR per night in major cities.
"""


async def upload(client: httpx.AsyncClient, tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    response = await client.post(
        "/documents", files={"file": (name, path.read_bytes(), "text/markdown")}
    )
    assert response.status_code == 202, response.text
    doc_id: str = response.json()["document"]["id"]
    assert (await client.get(f"/documents/{doc_id}")).json()["status"] == "ready"
    return doc_id


async def search(client: httpx.AsyncClient, query: str, **body: Any) -> dict[str, Any]:
    response = await client.post("/search", json={"query": query, **body})
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def top_heading(result: dict[str, Any]) -> list[str] | None:
    return result["hits"][0]["heading_path"] if result["hits"] else None


@pytest.fixture
async def corpus(client: httpx.AsyncClient, tmp_path: Path) -> dict[str, str]:
    return {
        "leave": await upload(client, tmp_path, "leave.md", LEAVE),
        "security": await upload(client, tmp_path, "security.md", SECURITY),
        "expenses": await upload(client, tmp_path, "expenses.md", EXPENSES),
    }


def add_user(
    sessions: sessionmaker[Session], tenant_id: uuid.UUID, email: str, role: Role = Role.EDITOR
) -> uuid.UUID:
    with sessions() as s, s.begin():
        user = User(tenant_id=tenant_id, email=email, role=role)
        s.add(user)
        s.flush()
        return user.id


async def test_exact_term_found_by_keyword_search(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    keyword = await search(client, "SEC-4471", mode="keyword")
    assert top_heading(keyword) == ["Security Standard", "Incident codes"]
    assert keyword["hits"][0]["keyword_rank"] == 1
    assert keyword["hits"][0]["vector_rank"] is None

    hybrid = await search(client, "SEC-4471")
    assert top_heading(hybrid) == ["Security Standard", "Incident codes"]


async def test_paraphrase_found_by_vector_search(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    query = "PTO allowance"  # neither word appears in any document
    assert (await search(client, query, mode="keyword"))["hits"] == []

    vector = await search(client, query, mode="vector")
    assert top_heading(vector) == ["Leave Policy", "Annual leave"]
    assert vector["embedding_model"] == "concept-embed-1"
    assert 0 < vector["hits"][0]["vector_similarity"] <= 1

    hybrid = await search(client, query)
    assert top_heading(hybrid) == ["Leave Policy", "Annual leave"]


async def test_natural_questions_match_any_word(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    result = await search(client, "How long must passwords be?", mode="keyword")
    assert top_heading(result) == ["Security Standard", "Credentials"]
    # Explicit syntax is applied strictly: this phrase does not occur anywhere.
    assert uses_search_syntax('"passwords are optional"')
    assert (await search(client, '"passwords are optional"', mode="keyword"))["hits"] == []


async def test_hybrid_fuses_both_rankings(
    client: httpx.AsyncClient, corpus: dict[str, str]
) -> None:
    result = await search(client, "hotel cost per night")
    hit = result["hits"][0]
    assert hit["heading_path"] == ["Expense Policy", "Travel"]
    assert (hit["keyword_rank"], hit["vector_rank"]) == (1, 1)
    assert hit["score"] == pytest.approx(2 / 61)
    assert hit["filename"] == "expenses.md"
    assert (hit["page_start"], hit["page_end"]) == (1, 1)
    assert set(result["timings_ms"]) >= {"keyword", "embed_query", "vector", "hydrate", "total"}


async def test_filters(client: httpx.AsyncClient, corpus: dict[str, str]) -> None:
    only_leave = await search(
        client, "passwords hotels vacation", filters={"document_ids": [corpus["leave"]]}
    )
    assert {h["document_id"] for h in only_leave["hits"]} == {corpus["leave"]}

    no_pdfs = await search(client, "passwords", filters={"file_types": ["pdf"]})
    assert no_pdfs["hits"] == []

    future = await search(client, "passwords", filters={"date_from": "2999-01-01T00:00:00Z"})
    assert future["hits"] == []


async def test_private_documents_never_retrieved_by_others(
    client: httpx.AsyncClient,
    corpus: dict[str, str],
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    alice = identity.user_id
    bob = add_user(sync_sessions, tenant_and_user[0], "bob@acme.test")
    with sync_sessions() as s, s.begin():
        other_tenant = Tenant(name="globex")
        s.add(other_tenant)
        s.flush()
        carol = User(tenant_id=other_tenant.id, email="carol@globex.test", role=Role.ADMIN)
        s.add(carol)
        s.flush()
        carol_id = carol.id

    queries = ["SEC-4471", "PTO allowance", "hotel cost per night"]
    for user in (bob, carol_id):
        identity.user_id = user
        for query in queries:
            for mode in ("keyword", "vector", "hybrid"):
                assert (await search(client, query, mode=mode))["hits"] == [], (user, query, mode)

    # Shared with the tenant: Bob can now find it; another tenant still cannot.
    with sync_sessions() as s, s.begin():
        s.execute(
            update(Document)
            .where(Document.id == uuid.UUID(corpus["security"]))
            .values(visibility=Visibility.TENANT)
        )
    identity.user_id = bob
    assert top_heading(await search(client, "SEC-4471")) == ["Security Standard", "Incident codes"]
    identity.user_id = carol_id
    assert (await search(client, "SEC-4471"))["hits"] == []
    identity.user_id = alice


async def test_identical_chunks_are_collapsed(client: httpx.AsyncClient, tmp_path: Path) -> None:
    boilerplate = "# Notice\n\nThis document is confidential and for internal use only.\n\n"
    await upload(client, tmp_path, "a.md", boilerplate + "# Pricing\n\nAlpha costs 10 EUR.\n")
    await upload(client, tmp_path, "b.md", boilerplate + "# Roadmap\n\nBeta ships in May.\n")

    collapsed = await search(client, "confidential internal use", mode="keyword")
    assert [h["heading_path"] for h in collapsed["hits"]] == [["Notice"]]
    assert collapsed["hits"][0]["duplicates"] == 1

    full = await search(
        client, "confidential internal use", mode="keyword", collapse_duplicates=False
    )
    assert len(full["hits"]) == 2


async def test_embedding_outage_degrades_to_keyword(
    client: httpx.AsyncClient, corpus: dict[str, str], embedder: FakeEmbeddingProvider
) -> None:
    embedder.fail = True
    result = await search(client, "SEC-4471")
    assert result["degraded"] == ["vector_unavailable"]
    assert top_heading(result) == ["Security Standard", "Incident codes"]
    assert result["hits"][0]["vector_rank"] is None

    response = await client.post("/search", json={"query": "SEC-4471", "mode": "vector"})
    assert response.status_code == 503
    assert response.json()["code"] == "vector_unavailable"


async def test_debug_endpoint_is_admin_only_and_shows_rankings(
    client: httpx.AsyncClient,
    corpus: dict[str, str],
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    body = {"query": "hotel cost per night", "rrf_k": 10, "vector_weight": 0.5}
    assert (await client.post("/admin/search/debug", json=body)).status_code == 403

    # Admins see the same documents they could read anyway: alice's are private to her.
    with sync_sessions() as s, s.begin():
        s.execute(update(User).where(User.id == identity.user_id).values(role=Role.ADMIN))
    debug = (await client.post("/admin/search/debug", json=body)).json()
    assert debug["keyword"][0]["rank"] == 1
    assert debug["vector"][0]["rank"] == 1
    top = debug["fused"][0]
    assert top["chunk_id"] == debug["hits"][0]["chunk_id"]
    assert top["score"] == pytest.approx(1 / 11 + 0.5 / 11)


def insert_closer_unreadable_chunks(
    sessions: sessionmaker[Session], query_vec: list[float], count: int = 2000
) -> None:
    """Chunks in another tenant, all nearer to the query than anything the user can read."""
    with sessions() as s, s.begin():
        other = Tenant(name="other-co")
        s.add(other)
        s.flush()
        owner = User(tenant_id=other.id, email="o@other.test", role=Role.EDITOR)
        s.add(owner)
        s.flush()
        doc = Document(
            tenant_id=other.id,
            owner_id=owner.id,
            filename="noise.md",
            file_type="md",
            mime_type="text/markdown",
            size_bytes=1,
            sha256="0" * 64,
            storage_uri="local://none",
            status="ready",
            meta={},
        )
        s.add(doc)
        s.flush()
        rng = np.random.default_rng(0)
        rows = []
        for i in range(count):
            v = np.array(query_vec) + rng.normal(0, 0.001, len(query_vec))
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "document_id": doc.id,
                    "tenant_id": other.id,
                    "chunk_index": i,
                    "text_hash": f"{i:064d}",
                    "embedding": str([float(x) for x in v / np.linalg.norm(v)]),
                }
            )
        s.execute(
            text(
                "INSERT INTO chunks (id, document_id, tenant_id, chunk_index, text, text_hash,"
                " page_start, page_end, char_start, char_end, heading_path, token_count,"
                " embedding, embedding_model) VALUES (:id, :document_id, :tenant_id,"
                " :chunk_index, 'noise', :text_hash, 1, 1, 0, 1, '[]', 1,"
                " CAST(:embedding AS vector), 'concept-embed-1')"
            ),
            rows,
        )
        s.execute(text("ANALYZE chunks"))


async def test_unreadable_nearer_chunks_do_not_hide_readable_ones(
    client: httpx.AsyncClient,
    tmp_path: Path,
    sync_sessions: sessionmaker[Session],
    embedder: FakeEmbeddingProvider,
) -> None:
    mine = await upload(client, tmp_path, "mine.md", LAPTOP_DOC)
    insert_closer_unreadable_chunks(sync_sessions, embedder.embed_query("lost stolen laptop"))
    result = await search(client, "lost stolen laptop", mode="vector", top_k=5)
    assert [h["document_id"] for h in result["hits"]] == [mine]


async def test_iterative_hnsw_scan_keeps_recall_under_access_filters(
    client: httpx.AsyncClient,
    tmp_path: Path,
    settings: Settings,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
    embedder: FakeEmbeddingProvider,
) -> None:
    """When Postgres walks the HNSW index, a plain scan returns only ~ef_search neighbours, all
    of which the access filter removes here. Iterative scans keep walking until enough rows pass.

    Small tables make the planner prefer an exact scan, so sorting is disabled to force the
    HNSW plan this test is about.
    """
    mine = await upload(client, tmp_path, "mine.md", LAPTOP_DOC)
    query_vec = embedder.embed_query("lost stolen laptop")
    insert_closer_unreadable_chunks(sync_sessions, query_vec)
    principal = Principal(user_id=identity.user_id, tenant_id=tenant_and_user[0])
    with sync_sessions() as s:
        closest_mine, farthest_noise = s.execute(
            text(
                "SELECT min(CASE WHEN tenant_id = :t THEN embedding <=> CAST(:q AS vector) END),"
                " max(CASE WHEN tenant_id <> :t THEN embedding <=> CAST(:q AS vector) END)"
                " FROM chunks"
            ),
            {"t": tenant_and_user[0], "q": str(query_vec)},
        ).one()
    assert farthest_noise < closest_mine  # every unreadable chunk is nearer than the readable one

    engine = create_async_engine(settings.database_url)
    try:
        async with AsyncSession(engine) as session:
            await session.execute(text("SET LOCAL enable_sort = off"))
            await session.execute(text("SET LOCAL hnsw.ef_search = 10"))
            plan = " | ".join(
                (
                    await session.execute(
                        text(
                            "EXPLAIN SELECT c.id FROM chunks c JOIN documents d"
                            " ON d.id = c.document_id WHERE d.tenant_id = :t"
                            " ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT 5"
                        ),
                        {"t": tenant_and_user[0], "q": str(query_vec)},
                    )
                ).scalars()
            )
            assert "ix_chunks_embedding_hnsw" in plan

            hits = await vector_search(
                session,
                principal,
                query_vec,
                model="concept-embed-1",
                limit=5,
                ef_search=10,
                filters=SearchFilters(),
            )
    finally:
        await engine.dispose()
    assert hits, "no results: the access filter removed every HNSW candidate"
    with sync_sessions() as s:
        doc_ids = {
            str(d)
            for d in s.scalars(
                select(Chunk.document_id).where(Chunk.id.in_([h.chunk_id for h in hits]))
            )
        }
    assert doc_ids == {mine}
