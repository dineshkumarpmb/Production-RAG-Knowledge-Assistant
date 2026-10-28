import json
import sys
import types
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.embeddings import EmbeddingError, get_embedding_provider
from app.embeddings.cache import embed_with_cache
from app.embeddings.local_bge import LocalEmbeddingProvider
from app.embeddings.voyage import VoyageEmbeddingProvider
from tests.conftest import FakeEmbeddingProvider

DIM = 4
Handler = Callable[[httpx.Request], httpx.Response]


def ok(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(len(t)), 0.0, 0.0, 1.0]} for i, t in enumerate(inputs)]
    data.reverse()  # the API does not promise order; the client must sort by index
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 7 * len(inputs)}})


class Recorder:
    def __init__(self, *responses: Handler) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"headers": request.headers, "json": json.loads(request.content)})
        handler = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return handler(request)


def voyage(recorder: Recorder, sleeps: list[float], **kwargs: Any) -> VoyageEmbeddingProvider:
    options: dict[str, Any] = {
        "api_key": "test-key",
        "model": "voyage-4",
        "dim": DIM,
        "price_per_million_tokens": 0.06,
        "requests_per_minute": 0,
        "max_retries": 2,
        "transport": httpx.MockTransport(recorder),
        "sleep": sleeps.append,
    }
    return VoyageEmbeddingProvider(**(options | kwargs))


def status(code: int, headers: dict[str, str] | None = None) -> Handler:
    return lambda _: httpx.Response(code, headers=headers, json={"detail": "nope"})


def test_batches_preserve_order_and_count_tokens() -> None:
    recorder, sleeps = Recorder(ok), []
    provider = voyage(recorder, sleeps, batch_size=2)
    result = provider.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [v[0] for v in result.vectors] == [1, 2, 3, 4, 5]
    assert result.tokens == 35
    assert [len(r["json"]["input"]) for r in recorder.requests] == [2, 2, 1]
    first = recorder.requests[0]
    assert first["headers"]["authorization"] == "Bearer test-key"
    assert first["json"] | {"input": None} == {
        "input": None,
        "model": "voyage-4",
        "input_type": "document",
        "output_dimension": DIM,
        "truncation": True,
    }


def test_batches_respect_token_budget() -> None:
    recorder = Recorder(ok)
    voyage(recorder, [], max_batch_tokens=10).embed_documents(["one two three four five six"] * 3)
    assert [len(r["json"]["input"]) for r in recorder.requests] == [1, 1, 1]


def test_query_uses_query_input_type() -> None:
    recorder = Recorder(ok)
    assert voyage(recorder, []).embed_query("hello") == [5.0, 0.0, 0.0, 1.0]
    assert recorder.requests[0]["json"]["input_type"] == "query"


def test_retries_rate_limits_using_retry_after() -> None:
    recorder, sleeps = Recorder(status(429, {"retry-after": "2"}), ok), []
    voyage(recorder, sleeps).embed_documents(["a"])
    assert len(recorder.requests) == 2
    assert sleeps == [2.0]


def test_retries_network_errors_with_backoff() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    calls = Recorder(broken, ok)
    sleeps: list[float] = []
    voyage(calls, sleeps).embed_documents(["a"])
    assert len(sleeps) == 1
    assert 0 <= sleeps[0] <= 1.0


def test_gives_up_after_max_retries() -> None:
    recorder, sleeps = Recorder(status(503)), []
    with pytest.raises(EmbeddingError, match="unavailable after 3 attempts"):
        voyage(recorder, sleeps).embed_documents(["a"])
    assert len(recorder.requests) == 3
    assert len(sleeps) == 2


def test_client_errors_are_not_retried() -> None:
    recorder = Recorder(status(401))
    with pytest.raises(EmbeddingError, match="HTTP 401"):
        voyage(recorder, []).embed_documents(["a"])
    assert len(recorder.requests) == 1


def test_wrong_dimension_rejected() -> None:
    with pytest.raises(EmbeddingError, match="not 8-dimensional"):
        voyage(Recorder(ok), [], dim=8).embed_documents(["a"])


def test_factory_requires_api_key(settings: Settings) -> None:
    for key in (None, ""):
        with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
            get_embedding_provider(settings.model_copy(update={"voyage_api_key": key}))


def install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    class SentenceTransformer:
        def __init__(self, name: str, device: str | None = None) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return dim

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            assert kwargs["normalize_embeddings"] is True
            return np.ones((len(texts), dim), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_sentence_transformers(monkeypatch, dim=DIM)
    provider = LocalEmbeddingProvider(model="BAAI/bge-m3", dim=DIM)
    result = provider.embed_documents(["hello world", "x"])
    assert result.vectors == [[1.0] * DIM, [1.0] * DIM]
    assert result.tokens == 3
    assert provider.embed_query("q") == [1.0] * DIM


def test_local_provider_checks_dimension_and_install(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_sentence_transformers(monkeypatch, dim=768)
    with pytest.raises(EmbeddingError, match="768-dim"):
        LocalEmbeddingProvider(model="other", dim=DIM)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingError, match="local' extra"):
        LocalEmbeddingProvider(model="BAAI/bge-m3", dim=DIM)


def test_cache_pays_once_per_text_and_model(sync_sessions: sessionmaker[Session]) -> None:
    provider = FakeEmbeddingProvider()
    texts = ["alpha", "beta", "alpha"]

    vectors, stats = embed_with_cache(sync_sessions, provider, texts)
    assert provider.embedded == ["alpha", "beta"]  # the repeated text is embedded once
    assert vectors[0] == vectors[2]
    assert (stats.inputs, stats.unique_inputs, stats.cache_hits, stats.embedded) == (3, 2, 0, 2)
    assert stats.cost_usd == pytest.approx(9 / 1_000_000)

    again, stats = embed_with_cache(sync_sessions, provider, ["beta", "alpha", "gamma"])
    assert provider.embedded == ["alpha", "beta", "gamma"]
    assert (stats.cache_hits, stats.embedded) == (2, 1)
    assert np.allclose(again[1], vectors[0])

    other_model = FakeEmbeddingProvider(model_name="fake-embed-2")
    embed_with_cache(sync_sessions, other_model, ["alpha"])
    assert other_model.embedded == ["alpha"]
