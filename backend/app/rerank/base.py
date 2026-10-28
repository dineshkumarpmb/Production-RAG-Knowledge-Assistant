from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class RerankError(Exception):
    """Reranking failed; retrieval falls back to the fused order."""


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]  # one per input document, in input order; higher = more relevant
    tokens: int  # billed tokens (estimated for local models)


class Reranker(Protocol):
    model_name: str
    price_per_million_tokens: float

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResult: ...
