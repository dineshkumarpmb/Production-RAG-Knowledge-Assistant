import threading

from app.core.config import Settings
from app.db.models import EMBEDDING_DIM
from app.embeddings.base import EmbeddingError, EmbeddingProvider, EmbeddingResult

__all__ = ["EmbeddingError", "EmbeddingProvider", "EmbeddingResult", "get_embedding_provider"]

DEFAULT_MODELS = {"voyage": "voyage-4", "local": "BAAI/bge-m3"}
DEFAULT_PRICES = {"voyage-4": 0.06, "voyage-4-lite": 0.02, "voyage-4-large": 0.12}

_providers: dict[tuple[str, str, str | None], EmbeddingProvider] = {}
_lock = threading.Lock()


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """One provider per process and model (local models are expensive to load)."""
    model = settings.embedding_model or DEFAULT_MODELS[settings.embedding_provider]
    key = (settings.embedding_provider, model, settings.model_server_url)
    with _lock:
        if key not in _providers:
            _providers[key] = _create(settings, model)
        return _providers[key]


def _create(settings: Settings, model: str) -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        if settings.model_server_url:
            from app.models_client import ServerEmbeddingProvider

            return ServerEmbeddingProvider(
                settings.model_server_url,
                expected_model=model,
                timeout_seconds=settings.model_server_timeout_seconds,
            )
        from app.embeddings.local_bge import LocalEmbeddingProvider

        return LocalEmbeddingProvider(
            model=model,
            dim=EMBEDDING_DIM,
            device=settings.local_model_device,
            dtype=settings.local_model_dtype,
        )

    from app.embeddings.voyage import VoyageEmbeddingProvider

    api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else ""
    if not api_key:
        raise EmbeddingError("VOYAGE_API_KEY is not set.")
    price = settings.embedding_price_per_million_tokens
    return VoyageEmbeddingProvider(
        api_key=api_key,
        model=model,
        dim=EMBEDDING_DIM,
        price_per_million_tokens=price if price is not None else DEFAULT_PRICES.get(model, 0.0),
        base_url=settings.voyage_base_url,
        batch_size=settings.embedding_batch_size,
        max_batch_tokens=settings.embedding_max_batch_tokens,
        max_retries=settings.embedding_max_retries,
        requests_per_minute=settings.embedding_requests_per_minute,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
