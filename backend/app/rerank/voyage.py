"""Voyage AI reranking (https://docs.voyageai.com/reference/reranker-api). Optional: the
project runs fully local by default."""

from collections.abc import Callable, Sequence
from typing import Any

import httpx

from app.embeddings.voyage import post_with_retries
from app.rerank.base import RerankError, RerankResult


class VoyageReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        price_per_million_tokens: float,
        base_url: str = "https://api.voyageai.com/v1",
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.model_name = model
        self.price_per_million_tokens = price_per_million_tokens
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        if not documents:
            return RerankResult([], 0)
        payload = {
            "query": query,
            "documents": list(documents),
            "model": self.model_name,
            "truncation": True,
        }
        body: dict[str, Any] = post_with_retries(
            self._client,
            "/rerank",
            payload,
            max_retries=self._max_retries,
            error=RerankError,
            service="Voyage rerank",
            sleep=self._sleep,
        )
        scores = [0.0] * len(documents)
        seen = set()
        for item in body["data"]:
            scores[item["index"]] = float(item["relevance_score"])
            seen.add(item["index"])
        if len(seen) != len(documents):
            raise RerankError(f"Voyage returned {len(seen)} scores for {len(documents)} documents.")
        return RerankResult(scores, int(body.get("usage", {}).get("total_tokens", 0)))
