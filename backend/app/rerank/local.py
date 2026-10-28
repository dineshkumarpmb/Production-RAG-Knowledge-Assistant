"""Local cross-encoder reranking with sentence-transformers (needs the `local` extra).

A cross-encoder reads the query and a chunk together, which is far more accurate than comparing
two separately computed embeddings, but too slow to run over the whole corpus; so it only
rescores the few dozen candidates hybrid search returns.

Models disagree on whether they output raw logits (ms-marco-MiniLM: roughly -11..+9) or
probabilities, so we always ask for raw logits and apply the sigmoid ourselves. Every local
reranker then scores 0..1 and a single threshold can mean "nothing relevant was found".
"""

import math
from collections.abc import Sequence
from typing import Any

from app.ingestion.tokens import count_tokens
from app.rerank.base import RerankError, RerankResult


class LocalReranker:
    price_per_million_tokens = 0.0

    def __init__(
        self,
        *,
        model: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankError(
                "Local reranking needs the 'local' extra: pip install '.[local]'."
            ) from exc

        self.model_name = model
        self._batch_size = batch_size
        options: dict[str, Any] = {"max_length": max_length, "device": device}
        if dtype != "float32":
            options["model_kwargs"] = {"dtype": dtype}
        self._model: Any = CrossEncoder(model, **options)
        self._raw_logits: Any = torch.nn.Identity()

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        if not documents:
            return RerankResult([], 0)
        logits = self._model.predict(
            [(query, doc) for doc in documents],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            activation_fn=self._raw_logits,
        )
        tokens = sum(count_tokens(query) + count_tokens(doc) for doc in documents)
        return RerankResult([_sigmoid(float(x)) for x in logits], tokens)


def _sigmoid(x: float) -> float:
    # Written to avoid overflow for very negative logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
