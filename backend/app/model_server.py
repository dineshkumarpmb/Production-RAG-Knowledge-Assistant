"""Local model server: loads the embedding model and reranker once and serves them over HTTP.

The API and the worker both need the models; loading them in each process would double the
memory (bge-m3 alone is ~2.3 GB in float32). Run with:

    uvicorn app.model_server:app --host 0.0.0.0 --port 8001
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

import structlog
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.errors import AppError, RequestIdMiddleware, register_exception_handlers
from app.core.logging import configure_logging
from app.db.models import EMBEDDING_DIM
from app.embeddings import DEFAULT_MODELS as EMBEDDING_MODELS
from app.embeddings.base import EmbeddingProvider
from app.rerank import DEFAULT_MODELS as RERANK_MODELS
from app.rerank.base import Reranker

log = structlog.get_logger(__name__)

MAX_TEXTS = 256
MAX_TEXT_CHARS = 20_000


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)
    input_type: Literal["document", "query"] = "document"


class EmbedResponse(BaseModel):
    model: str
    vectors: list[list[float]]
    tokens: int


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    documents: list[str] = Field(max_length=MAX_TEXTS)


class RerankResponse(BaseModel):
    model: str
    scores: list[float]
    tokens: int


class Info(BaseModel):
    embedding_model: str
    dim: int
    reranker_model: str | None
    dtype: str
    device: str | None
    load_seconds: float


class Models:
    """Loaded models. One lock: a single model call already uses every CPU core, and running
    several at once would only multiply peak memory."""

    def __init__(self, settings: Settings) -> None:
        from app.embeddings.local_bge import LocalEmbeddingProvider
        from app.rerank.local import LocalReranker

        started = time.perf_counter()
        self.settings = settings
        self.lock = threading.Lock()
        self.embedder: EmbeddingProvider = LocalEmbeddingProvider(
            model=settings.embedding_model or EMBEDDING_MODELS["local"],
            dim=EMBEDDING_DIM,
            device=settings.local_model_device,
            dtype=settings.local_model_dtype,
        )
        self.reranker: Reranker | None = None
        if settings.reranker_provider == "local":
            self.reranker = LocalReranker(
                model=settings.reranker_model or RERANK_MODELS["local"],
                max_length=settings.rerank_max_length,
                device=settings.local_model_device,
                dtype=settings.local_model_dtype,
            )
        self.load_seconds = round(time.perf_counter() - started, 1)

    def run[T](self, work: Callable[[], T]) -> T:
        with self.lock:
            return work()


def create_model_app(load: Callable[[Settings], Any] = Models) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("models.loading", dtype=settings.local_model_dtype)
        state["models"] = await asyncio.to_thread(load, settings)
        log.info("models.ready", load_seconds=getattr(state["models"], "load_seconds", None))
        yield

    app = FastAPI(title="RAG model server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    def models() -> Any:
        if "models" not in state:
            raise AppError("models_loading", "Models are still loading.", 503)
        return state["models"]

    def check_lengths(texts: list[str]) -> None:
        if any(len(t) > MAX_TEXT_CHARS for t in texts):
            raise AppError(
                "text_too_long", f"Texts are limited to {MAX_TEXT_CHARS} characters.", 422
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        models()  # 503 until loaded, so Compose can wait for readiness
        return {"status": "ok"}

    @app.get("/info")
    async def info() -> Info:
        m = models()
        return Info(
            embedding_model=m.embedder.model_name,
            dim=m.embedder.dim,
            reranker_model=m.reranker.model_name if m.reranker else None,
            dtype=settings.local_model_dtype,
            device=settings.local_model_device,
            load_seconds=m.load_seconds,
        )

    @app.post("/embed")
    async def embed(body: EmbedRequest) -> EmbedResponse:
        m = models()
        check_lengths(body.texts)
        if body.input_type == "query" and len(body.texts) == 1:
            vector = await run_in_threadpool(m.run, lambda: m.embedder.embed_query(body.texts[0]))
            return EmbedResponse(model=m.embedder.model_name, vectors=[vector], tokens=0)
        result = await run_in_threadpool(m.run, lambda: m.embedder.embed_documents(body.texts))
        return EmbedResponse(
            model=m.embedder.model_name, vectors=result.vectors, tokens=result.tokens
        )

    @app.post("/rerank")
    async def rerank(body: RerankRequest) -> RerankResponse:
        m = models()
        if m.reranker is None:
            raise AppError("reranker_disabled", "This model server has no reranker loaded.", 404)
        check_lengths(body.documents)
        result = await run_in_threadpool(
            m.run, lambda: m.reranker.rerank(body.query, body.documents)
        )
        return RerankResponse(
            model=m.reranker.model_name, scores=result.scores, tokens=result.tokens
        )

    return app


app = create_model_app()
