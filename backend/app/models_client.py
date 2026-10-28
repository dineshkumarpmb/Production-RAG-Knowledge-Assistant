"""Clients for the local model server (app.model_server), used by the API and the worker."""

from collections.abc import Callable, Sequence
from typing import Any

import httpx

from app.embeddings.base import EmbeddingError, EmbeddingResult
from app.embeddings.voyage import post_with_retries
from app.rerank.base import RerankError, RerankResult

_EMBED_BATCH = 64  # texts per request; keeps request bodies and peak memory moderate
_MAX_RETRIES = 3


class _ModelServer:
    def __init__(
        self,
        base_url: str,
        *,
        error: type[Exception],
        timeout_seconds: float,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None] | None,
        client: httpx.Client | None,
    ) -> None:
        self._error = error
        self._sleep = sleep
        self._client = client or httpx.Client(
            base_url=base_url, timeout=timeout_seconds, transport=transport
        )
        self.info = self._get_info()

    def _get_info(self) -> dict[str, Any]:
        try:
            response = self._client.get("/info")
        except httpx.TransportError as exc:
            raise self._error(f"Model server unreachable ({type(exc).__name__}).") from exc
        if response.status_code != 200:
            raise self._error(f"Model server not ready (HTTP {response.status_code}).")
        info: dict[str, Any] = response.json()
        return info

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return post_with_retries(
            self._client,
            path,
            payload,
            max_retries=_MAX_RETRIES,
            error=self._error,
            service="Model server",
            sleep=self._sleep,
        )


class ServerEmbeddingProvider:
    price_per_million_tokens = 0.0

    def __init__(
        self,
        base_url: str,
        *,
        expected_model: str,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        client: httpx.Client | None = None,  # e.g. a TestClient in tests
    ) -> None:
        self._server = _ModelServer(
            base_url,
            error=EmbeddingError,
            timeout_seconds=timeout_seconds,
            transport=transport,
            sleep=sleep,
            client=client,
        )
        self.model_name: str = self._server.info["embedding_model"]
        self.dim: int = self._server.info["dim"]
        if self.model_name != expected_model:
            raise EmbeddingError(
                f"Model server runs {self.model_name}, but {expected_model} is configured."
            )

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), _EMBED_BATCH):
            body = self._server.post(
                "/embed", {"texts": list(texts[i : i + _EMBED_BATCH]), "input_type": "document"}
            )
            vectors.extend(body["vectors"])
            tokens += body["tokens"]
        return EmbeddingResult(vectors, tokens)

    def embed_query(self, text: str) -> list[float]:
        body = self._server.post("/embed", {"texts": [text], "input_type": "query"})
        vector: list[float] = body["vectors"][0]
        return vector


class ServerReranker:
    price_per_million_tokens = 0.0

    def __init__(
        self,
        base_url: str,
        *,
        expected_model: str,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        client: httpx.Client | None = None,  # e.g. a TestClient in tests
    ) -> None:
        self._server = _ModelServer(
            base_url,
            error=RerankError,
            timeout_seconds=timeout_seconds,
            transport=transport,
            sleep=sleep,
            client=client,
        )
        model = self._server.info.get("reranker_model")
        if model is None:
            raise RerankError("The model server has no reranker loaded.")
        if model != expected_model:
            raise RerankError(f"Model server runs {model}, but {expected_model} is configured.")
        self.model_name: str = model

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        if not documents:
            return RerankResult([], 0)
        body = self._server.post("/rerank", {"query": query, "documents": list(documents)})
        return RerankResult([float(s) for s in body["scores"]], int(body["tokens"]))
