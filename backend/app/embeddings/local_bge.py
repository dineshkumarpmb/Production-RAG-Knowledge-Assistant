"""Local embeddings with sentence-transformers (default BAAI/bge-m3). Free, but needs the
`local` extra (`pip install ".[local]"`, which pulls in PyTorch) and ideally a GPU."""

from collections.abc import Sequence
from typing import Any

from app.embeddings.base import EmbeddingError, EmbeddingResult
from app.ingestion.tokens import count_tokens


class LocalEmbeddingProvider:
    price_per_million_tokens = 0.0

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        batch_size: int = 16,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "Local embeddings need the 'local' extra: pip install '.[local]'."
            ) from exc

        self.model_name = model
        self.dim = dim
        self._batch_size = batch_size
        options: dict[str, Any] = {"device": device}
        if dtype != "float32":  # half precision halves memory; see README for the trade-off
            options["model_kwargs"] = {"dtype": dtype}
        self._model: Any = SentenceTransformer(model, **options)
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        actual = get_dim()
        if actual != dim:
            raise EmbeddingError(f"{model} produces {actual}-dim vectors; expected {dim}.")

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(self._encode(texts), sum(count_tokens(t) for t in texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]  # bge-m3 needs no query instruction for dense retrieval
