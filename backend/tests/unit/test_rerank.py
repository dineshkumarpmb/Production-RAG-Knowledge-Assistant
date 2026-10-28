import json
import math
import sys
import types
import uuid
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.embeddings import EmbeddingError
from app.model_server import create_model_app
from app.models_client import ServerEmbeddingProvider, ServerReranker
from app.rerank import RerankError, get_reranker
from app.rerank.local import LocalReranker
from app.rerank.voyage import VoyageReranker
from app.retrieval.context import assemble_context, merge_overlapping
from app.retrieval.types import SearchHit
from tests.conftest import FakeEmbeddingProvider, FakeReranker

# --- context assembly -------------------------------------------------------------------

DOC_A, DOC_B = uuid.UUID(int=1), uuid.UUID(int=2)


def hit(doc: uuid.UUID, index: int, text: str, score: float, page: int = 1, **kw: Any) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid5(doc, str(index)),
        document_id=doc,
        document_title="Doc A" if doc == DOC_A else "Doc B",
        filename="a.pdf" if doc == DOC_A else "b.pdf",
        chunk_index=index,
        page_start=page,
        page_end=page,
        heading_path=kw.get("path", ["Policy"]),
        text=text,
        score=0.01,
        keyword_rank=None,
        vector_rank=None,
        keyword_score=None,
        vector_similarity=None,
        duplicates=0,
        fused_rank=1,
        rerank_score=score,
    )


OVERLAP = "Shared sentence carried over between the two chunks."


def test_merge_overlapping_drops_shared_text_once() -> None:
    first = f"Opening paragraph.\n\n{OVERLAP}"
    second = f"{OVERLAP}\n\nClosing paragraph."
    assert (
        merge_overlapping(first, second) == f"Opening paragraph.\n\n{OVERLAP}\n\nClosing paragraph."
    )


def test_merge_ignores_coincidental_short_matches() -> None:
    assert merge_overlapping("ends with the", "e start") == "ends with the\n\ne start"
    assert merge_overlapping("alpha beta", "beta gamma") == "alpha beta\n\nbeta gamma"


def test_adjacent_chunks_merge_into_one_passage() -> None:
    hits = [
        hit(DOC_A, 4, f"{OVERLAP} Part two.", 0.9, page=3, path=["Policy", "Leave"]),
        hit(DOC_B, 0, "Other document.", 0.5),
        hit(DOC_A, 3, f"Part one. {OVERLAP}", 0.7, page=2, path=["Policy", "Leave", "Annual"]),
        hit(DOC_A, 9, "Far away chunk.", 0.2, page=7),
    ]
    blocks = assemble_context(hits, token_budget=10_000)

    assert [b.filename for b in blocks] == ["a.pdf", "b.pdf", "a.pdf"]
    merged = blocks[0]
    assert merged.text == f"Part one. {OVERLAP} Part two."
    assert (merged.page_start, merged.page_end) == (2, 3)
    assert merged.chunk_ids == [hits[2].chunk_id, hits[0].chunk_id]  # document order
    assert merged.heading_path == ["Policy", "Leave"]  # shared prefix
    assert merged.score == 0.9
    assert blocks[2].text == "Far away chunk."


def test_token_budget_skips_blocks_that_do_not_fit() -> None:
    long = " ".join(["word"] * 300)
    hits = [hit(DOC_A, 0, long, 0.9), hit(DOC_B, 0, "Short answer.", 0.5), hit(DOC_B, 5, long, 0.4)]
    blocks = assemble_context(hits, token_budget=310)
    assert [b.score for b in blocks] == [0.9, 0.5]  # the second long block no longer fits
    blocks = assemble_context(hits, token_budget=200)
    assert [b.text for b in blocks] == ["Short answer."]  # the long top block is skipped
    assert assemble_context([], token_budget=100) == []


# --- local cross-encoder ----------------------------------------------------------------


def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
    class CrossEncoder:
        def __init__(self, name: str, **kwargs: Any) -> None:
            seen["init"] = (name, kwargs)

        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
            seen["pairs"] = pairs
            seen["activation_fn"] = kwargs["activation_fn"]
            return np.array([float(len(doc)) - 10 for _, doc in pairs], dtype=np.float32)  # logits

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Identity=lambda: "identity")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_local_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    fake_cross_encoder(monkeypatch, seen)
    reranker = LocalReranker(model="BAAI/bge-reranker-v2-m3", max_length=256, dtype="float16")
    assert seen["init"] == (
        "BAAI/bge-reranker-v2-m3",
        {"max_length": 256, "device": None, "model_kwargs": {"dtype": "float16"}},
    )
    result = reranker.rerank("hotel limit", ["short", "a much longer document"])
    # Raw logits (-5 and +12) are requested and squashed to 0..1 with a sigmoid.
    assert seen["activation_fn"] == "identity"
    assert result.scores == pytest.approx([1 / (1 + math.exp(5)), 1 / (1 + math.exp(-12))])
    assert seen["pairs"] == [("hotel limit", "short"), ("hotel limit", "a much longer document")]
    assert result.tokens > 0
    assert reranker.rerank("q", []).scores == []
    extremes = reranker.rerank("q", ["x" * 5000, ""]).scores  # logits +4990 and -10: no overflow
    assert extremes[0] == 1.0 and 0 < extremes[1] < 1


def test_local_reranker_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankError, match="local' extra"):
        LocalReranker(model="x")


# --- Voyage rerank ----------------------------------------------------------------------


def test_voyage_reranker_maps_scores_back_to_input_order() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ],
                "usage": {"total_tokens": 40},
            },
        )

    reranker = VoyageReranker(
        api_key="k",
        model="rerank-2.5",
        price_per_million_tokens=0.05,
        transport=httpx.MockTransport(handler),
    )
    result = reranker.rerank("q", ["first", "second"])
    assert result.scores == [0.2, 0.9]
    assert result.tokens == 40
    assert requests[0] == {
        "query": "q",
        "documents": ["first", "second"],
        "model": "rerank-2.5",
        "truncation": True,
    }


def test_voyage_reranker_errors() -> None:
    def partial(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "relevance_score": 1.0}]})

    def unauthorised(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    for handler, message in ((partial, "1 scores for 2"), (unauthorised, "HTTP 401")):
        reranker = VoyageReranker(
            api_key="k",
            model="rerank-2.5",
            price_per_million_tokens=0,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(RerankError, match=message):
            reranker.rerank("q", ["a", "b"])


def test_reranker_factory(settings: Settings) -> None:
    assert get_reranker(settings.model_copy(update={"reranker_provider": "none"})) is None
    with pytest.raises(RerankError, match="VOYAGE_API_KEY"):
        get_reranker(settings.model_copy(update={"reranker_provider": "voyage"}))


# --- model server and its clients -------------------------------------------------------


class FakeModels:
    load_seconds = 0.1

    def __init__(self, with_reranker: bool = True) -> None:
        self.embedder = FakeEmbeddingProvider(model_name="BAAI/bge-m3")
        self.reranker = FakeReranker() if with_reranker else None

    def run(self, work: Any) -> Any:
        return work()


@pytest.fixture
def model_server() -> TestClient:
    return TestClient(create_model_app(load=lambda _: FakeModels()))


def test_model_server_endpoints(model_server: TestClient) -> None:
    with model_server as client:
        info = client.get("/info").json()
        assert (info["embedding_model"], info["dim"], info["reranker_model"]) == (
            "BAAI/bge-m3",
            1024,
            "fake-rerank-1",
        )
        assert client.get("/healthz").json() == {"status": "ok"}

        vectors = client.post("/embed", json={"texts": ["a", "b"]}).json()["vectors"]
        assert len(vectors) == 2 and len(vectors[0]) == 1024
        query = client.post("/embed", json={"texts": ["a"], "input_type": "query"}).json()
        assert query["vectors"][0] == pytest.approx(vectors[0])

        scores = client.post(
            "/rerank", json={"query": "hotel limit", "documents": ["hotel limit", "nothing"]}
        ).json()["scores"]
        assert scores == [1.0, 0.0]

        too_long = client.post("/embed", json={"texts": ["x" * 20_001]})
        assert too_long.status_code == 422
        assert client.post("/embed", json={"texts": []}).status_code == 422


def test_model_server_reports_loading() -> None:
    app = create_model_app(load=lambda _: FakeModels())
    client = TestClient(app)  # not entered: lifespan has not run, so nothing is loaded
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["code"] == "models_loading"


def test_clients_talk_to_the_model_server(model_server: TestClient) -> None:
    with model_server as client:
        embedder = ServerEmbeddingProvider(
            "http://testserver", expected_model="BAAI/bge-m3", client=client
        )
        assert (embedder.model_name, embedder.dim) == ("BAAI/bge-m3", 1024)
        result = embedder.embed_documents([f"text {i}" for i in range(70)])  # two batches
        assert len(result.vectors) == 70
        assert len(embedder.embed_query("q")) == 1024

        reranker = ServerReranker(
            "http://testserver", expected_model="fake-rerank-1", client=client
        )
        assert reranker.rerank("hotel", ["hotel", "x"]).scores == [1.0, 0.0]

        with pytest.raises(EmbeddingError, match="runs BAAI/bge-m3, but other is configured"):
            ServerEmbeddingProvider("http://testserver", expected_model="other", client=client)


def test_clients_report_missing_reranker_and_unreachable_server() -> None:
    no_reranker = TestClient(create_model_app(load=lambda _: FakeModels(with_reranker=False)))
    with no_reranker as client, pytest.raises(RerankError, match="no reranker"):
        ServerReranker("http://testserver", expected_model="x", client=client)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(EmbeddingError, match="unreachable"):
        ServerEmbeddingProvider(
            "http://models:8001", expected_model="m", transport=httpx.MockTransport(refuse)
        )
