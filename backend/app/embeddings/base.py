from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class EmbeddingError(Exception):
    """Embedding failed. The message is stored on the document, so keep it free of secrets."""


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    tokens: int  # as billed by the provider (estimated for local models)


class EmbeddingProvider(Protocol):
    model_name: str
    dim: int
    price_per_million_tokens: float

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def embed_query(self, text: str) -> list[float]: ...
