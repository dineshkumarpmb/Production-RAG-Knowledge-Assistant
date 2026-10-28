"""Reranking and context assembly against real PostgreSQL (skipped without TEST_DATABASE_URL)."""

import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Role, User
from app.retrieval.service import RetrievalConfig
from tests.conftest import FakeEmbeddingProvider, FakeReranker, Identity, requires_postgres
from tests.postgres.test_retrieval import ConceptEmbedder

pytestmark = requires_postgres


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return ConceptEmbedder()


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(candidates=50, ef_search=100, rerank_candidates=30, rerank_min_score=0.6)


# Keyword search ranks REPEATS first (the word appears ten times); the word-overlap reranker
# prefers BOTH, which contains every query word once.
REPEATS = "# Zebras\n\n" + "Zebra zebra zebra zebra zebra zebra zebra zebra zebra zebra.\n"
BOTH = "# Mixed\n\nA zebra met a quokka near the river.\n"
QUOKKAS = "# Quokka Habitats\n\n" + "\n\n".join(
    f"Paragraph {i} describes quokka habitats on Rottnest Island in some detail, covering "
    f"vegetation, water sources, shelter from heat and the seasonal patterns observed."
    for i in range(40)
)


async def upload(client: httpx.AsyncClient, tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    response = await client.post(
        "/documents", files={"file": (name, path.read_bytes(), "text/markdown")}
    )
    assert response.status_code == 202, response.text
    doc_id: str = response.json()["document"]["id"]
    return doc_id


async def post(client: httpx.AsyncClient, path: str, **body: Any) -> dict[str, Any]:
    response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


async def test_rerank_reorders_fused_results(
    client: httpx.AsyncClient, tmp_path: Path, reranker: FakeReranker
) -> None:
    repeats = await upload(client, tmp_path, "repeats.md", REPEATS)
    both = await upload(client, tmp_path, "both.md", BOTH)

    plain = await post(client, "/search", query="zebra quokka", mode="keyword")
    assert [h["document_id"] for h in plain["hits"]] == [repeats, both]
    assert plain["reranker_model"] is None
    assert all(h["rerank_score"] is None for h in plain["hits"])

    reranked = await post(client, "/search", query="zebra quokka", mode="keyword", rerank=True)
    assert [h["document_id"] for h in reranked["hits"]] == [both, repeats]
    assert [h["fused_rank"] for h in reranked["hits"]] == [2, 1]
    assert [h["rerank_score"] for h in reranked["hits"]] == [1.0, 0.5]
    assert reranked["reranker_model"] == "fake-rerank-1"
    assert "rerank" in reranked["timings_ms"]
    assert not reranked["not_found"]

    # The reranker sees where each chunk comes from, not just its text.
    query, documents = reranker.calls[-1]
    assert query == "zebra quokka"
    assert documents[0].startswith("Zebras > Zebras\n\n")  # title > heading path


async def test_low_rerank_scores_mean_not_found(client: httpx.AsyncClient, tmp_path: Path) -> None:
    await upload(client, tmp_path, "repeats.md", REPEATS)
    weak = await post(client, "/search", query="zebra unicorn", mode="keyword", rerank=True)
    assert weak["hits"]  # something matched...
    assert weak["hits"][0]["rerank_score"] == 0.5
    assert weak["not_found"]  # ...but nothing scored above RERANK_MIN_SCORE (0.6)

    retrieved = await post(client, "/retrieve", query="zebra unicorn")
    assert retrieved["not_found"]
    assert retrieved["blocks"] == []


async def test_reranker_outage_keeps_fused_order(
    client: httpx.AsyncClient, tmp_path: Path, reranker: FakeReranker
) -> None:
    repeats = await upload(client, tmp_path, "repeats.md", REPEATS)
    both = await upload(client, tmp_path, "both.md", BOTH)
    reranker.fail = True

    result = await post(client, "/search", query="zebra quokka", mode="keyword", rerank=True)
    assert result["degraded"] == ["rerank_unavailable"]
    assert [h["document_id"] for h in result["hits"]] == [repeats, both]
    assert result["reranker_model"] is None
    assert not result["not_found"]  # the threshold only applies to reranked scores


async def test_retrieve_merges_adjacent_chunks_into_passages(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    doc_id = await upload(client, tmp_path, "quokkas.md", QUOKKAS)
    chunks = (await client.get(f"/documents/{doc_id}/chunks")).json()
    assert chunks["total"] >= 3  # a long section split into overlapping chunks

    result = await post(
        client, "/retrieve", query="quokka habitats Rottnest", top_k=10, token_budget=10_000
    )
    assert not result["not_found"]
    (block,) = result["blocks"]
    assert len(block["chunk_ids"]) == chunks["total"]
    assert block["heading_path"] == ["Quokka Habitats"]
    assert (block["page_start"], block["page_end"]) == (1, 1)
    # Overlapping paragraphs appear once in the merged passage.
    for i in range(40):
        assert block["text"].count(f"Paragraph {i} describes") == 1
    assert block["token_count"] > 0
    assert result["reranker_model"] == "fake-rerank-1"
    assert result["rerank_cost_usd"] > 0


async def test_retrieve_respects_the_token_budget(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    await upload(client, tmp_path, "quokkas.md", QUOKKAS)
    await upload(client, tmp_path, "both.md", BOTH)
    result = await post(client, "/retrieve", query="quokka zebra river", token_budget=100)
    assert sum(b["token_count"] for b in result["blocks"]) <= 100
    assert [b["filename"] for b in result["blocks"]] == ["both.md"]


async def test_retrieve_never_returns_unreadable_documents(
    client: httpx.AsyncClient,
    tmp_path: Path,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    await upload(client, tmp_path, "both.md", BOTH)
    with sync_sessions() as s, s.begin():
        bob = User(tenant_id=tenant_and_user[0], email="bob@acme.test", role=Role.EDITOR)
        s.add(bob)
        s.flush()
        identity.user_id = bob.id
    result = await post(client, "/retrieve", query="zebra quokka river")
    assert result["blocks"] == []
    assert result["not_found"]
