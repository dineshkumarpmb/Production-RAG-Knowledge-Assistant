"""The retrieval evaluation harness end to end on the sample corpus (PostgreSQL only).

The concept embedder is a stand-in, so these numbers check the plumbing, not search quality.
"""

import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.access import Principal
from app.evaluation.retrieval import CONFIGS, load_dataset, render_markdown
from app.evaluation.retrieval import evaluate as run_evaluation
from app.retrieval.service import HybridRetriever, RetrievalConfig
from tests.conftest import FakeEmbeddingProvider, FakeReranker, Identity, requires_postgres
from tests.postgres.test_retrieval import ConceptEmbedder

pytestmark = requires_postgres

EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return ConceptEmbedder()


async def test_harness_on_sample_corpus(
    client: httpx.AsyncClient,
    settings: Settings,
    identity: Identity,
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
    embedder: FakeEmbeddingProvider,
    reranker: FakeReranker,
) -> None:
    for path in sorted((EVAL_DIR / "corpus" / "sample").glob("*.md")):
        response = await client.post(
            "/documents", files={"file": (path.name, path.read_bytes(), "text/markdown")}
        )
        assert response.status_code == 202, response.text

    questions = load_dataset(EVAL_DIR / "datasets" / "sample_qa.jsonl")
    engine = create_async_engine(settings.database_url)
    try:
        retriever = HybridRetriever(
            async_sessionmaker(engine, expire_on_commit=False),
            lambda: embedder,
            RetrievalConfig(),
            reranker_factory=lambda: reranker,
        )
        report = await run_evaluation(
            questions,
            retriever,
            {"alice@acme.test": Principal(identity.user_id, tenant_and_user[0])},
            default_email="alice@acme.test",
            configs=list(CONFIGS.values()),
            ks=(1, 5),
            dataset_name="sample_qa.jsonl",
        )
    finally:
        await engine.dispose()

    assert (report.answerable, report.unanswerable) == (12, 2)
    by_config = {s.config: s for s in report.summaries}
    assert set(by_config) == {"hybrid", "keyword", "vector", "hybrid+rerank"}
    assert all(s.questions == 12 for s in report.summaries)
    # Exact codes are what keyword search is for.
    assert by_config["keyword"].by_type["exact-term"]["recall@1"] == 1.0
    assert by_config["hybrid"].by_type["exact-term"]["recall@5"] == 1.0
    assert len(report.results) == 48

    reranked = by_config["hybrid+rerank"]
    assert reranked.rerank_p50_ms is not None
    assert by_config["hybrid"].rerank_p50_ms is None
    assert [t["threshold"] for t in report.thresholds] == [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    assert all(0 <= t["unanswerable_refused"] <= 1 for t in report.thresholds)

    markdown = render_markdown(report, (1, 5))
    assert "| hybrid+rerank |" in markdown
    assert "RERANK_MIN_SCORE" in markdown
    assert "Recall@5 by question type" in markdown
