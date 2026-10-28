import threading

from app.core.config import Settings
from app.rerank.base import Reranker, RerankError, RerankResult

__all__ = ["RerankError", "RerankResult", "Reranker", "get_reranker"]

# MiniLM-L6 (22M parameters, English) fits next to bge-m3 on a small laptop;
# BAAI/bge-reranker-v2-m3 (568M, multilingual) is the stronger choice with ~4 GB more RAM.
DEFAULT_MODELS = {"local": "cross-encoder/ms-marco-MiniLM-L-6-v2", "voyage": "rerank-2.5"}
DEFAULT_PRICES = {"rerank-2.5": 0.05, "rerank-2.5-lite": 0.02}

_rerankers: dict[tuple[str, str, str | None], Reranker] = {}
_lock = threading.Lock()


def get_reranker(settings: Settings) -> Reranker | None:
    """The configured reranker (one per process), or None when reranking is switched off."""
    if settings.reranker_provider == "none":
        return None
    model = settings.reranker_model or DEFAULT_MODELS[settings.reranker_provider]
    key = (settings.reranker_provider, model, settings.model_server_url)
    with _lock:
        if key not in _rerankers:
            _rerankers[key] = _create(settings, model)
        return _rerankers[key]


def _create(settings: Settings, model: str) -> Reranker:
    if settings.reranker_provider == "local":
        if settings.model_server_url:
            from app.models_client import ServerReranker

            return ServerReranker(
                settings.model_server_url,
                expected_model=model,
                timeout_seconds=settings.model_server_timeout_seconds,
            )
        from app.rerank.local import LocalReranker

        return LocalReranker(
            model=model,
            max_length=settings.rerank_max_length,
            device=settings.local_model_device,
            dtype=settings.local_model_dtype,
        )

    from app.rerank.voyage import VoyageReranker

    api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else ""
    if not api_key:
        raise RerankError("VOYAGE_API_KEY is not set.")
    price = settings.reranker_price_per_million_tokens
    return VoyageReranker(
        api_key=api_key,
        model=model,
        price_per_million_tokens=price if price is not None else DEFAULT_PRICES.get(model, 0.0),
        base_url=settings.voyage_base_url,
    )
