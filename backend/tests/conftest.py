import asyncio
import hashlib
import os
import re
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_current_user
from app.api.routes.chat import get_answer_generator, get_conversation_helper
from app.api.routes.search import get_retriever
from app.core.config import Settings, get_settings
from app.core.ratelimit import MemoryRateLimiter, get_rate_limiter
from app.db.access import RlsContext, rls_session_info
from app.db.base import Base
from app.db.models import EMBEDDING_DIM, Role, Tenant, User
from app.db.session import get_async_sessionmaker, get_db, sync_url
from app.embeddings import EmbeddingError, EmbeddingResult
from app.generation.citations import CitationResolver
from app.generation.claude import LLMError
from app.generation.conversation import HelperResult
from app.generation.prompts import HistoryMessage
from app.generation.types import (
    CitationAdded,
    Completed,
    GenerationEvent,
    Span,
    TextDelta,
    Usage,
    UsageUpdate,
)
from app.ingestion.pipeline import ingest_document
from app.main import create_app
from app.rerank import RerankError, RerankResult
from app.retrieval.context import ContextBlock
from app.retrieval.service import HybridRetriever, RetrievalConfig
from app.storage.local import LocalStorage, get_storage
from app.workers.queue import get_ingestion_queue

# Set to run the suite against PostgreSQL, e.g.
#   TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5433/rag_test
# The database is wiped. Without it, tests use SQLite and Postgres-only tests are skipped.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
BACKEND_DIR = Path(__file__).resolve().parents[1]

requires_postgres = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="needs PostgreSQL (set TEST_DATABASE_URL)"
)


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    # psycopg's async mode cannot use Windows' default Proactor event loop.
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


@pytest.fixture(scope="session")
def postgres_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str | None]:
    """Fresh schema built by the real migrations, once per test session."""
    if not TEST_DATABASE_URL:
        yield None
        return
    engine = create_engine(sync_url(TEST_DATABASE_URL))
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()

    previous = os.environ.get("MIGRATION_DATABASE_URL")
    os.environ["MIGRATION_DATABASE_URL"] = TEST_DATABASE_URL
    get_settings.cache_clear()
    try:
        config = Config(str(BACKEND_DIR / "alembic.ini"))
        config.set_main_option("script_location", str(BACKEND_DIR / "app" / "db" / "migrations"))
        config.attributes["configure_logging"] = False
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("MIGRATION_DATABASE_URL", None)
        else:
            os.environ["MIGRATION_DATABASE_URL"] = previous
        get_settings.cache_clear()
    yield TEST_DATABASE_URL


def app_role_url(owner_url: str) -> str:
    """The same database as the unprivileged `rag_app` role, so row-level security applies."""
    password = Settings(_env_file=None).app_db_password.get_secret_value()  # type: ignore[call-arg]
    url = make_url(owner_url).set(username="rag_app", password=password)
    return url.render_as_string(hide_password=False)


@pytest.fixture
def settings(tmp_path: Path, postgres_url: str | None) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        env="test",
        log_json=False,
        log_level="WARNING",
        database_url=(
            app_role_url(postgres_url)
            if postgres_url
            else f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"
        ),
        migration_database_url=postgres_url,
        storage_dir=tmp_path / "storage",
        max_upload_bytes=1024 * 1024,
    )


@pytest.fixture
def storage(settings: Settings) -> LocalStorage:
    return LocalStorage(settings.storage_dir)


@pytest.fixture
def sync_sessions(settings: Settings, postgres_url: str | None) -> Iterator[sessionmaker[Session]]:
    """Sessions like the ingestion worker's: the app role with system RLS scope."""
    engine = create_engine(sync_url(settings.database_url))
    if postgres_url:
        owner = create_engine(sync_url(postgres_url))
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        with owner.begin() as conn:
            conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
        owner.dispose()
    else:
        Base.metadata.create_all(engine)
    yield sessionmaker(engine, expire_on_commit=False, info=rls_session_info(RlsContext.system()))
    engine.dispose()


@pytest.fixture
def tenant_and_user(sync_sessions: sessionmaker[Session]) -> tuple[uuid.UUID, uuid.UUID]:
    with sync_sessions() as s, s.begin():
        tenant = Tenant(name="acme")
        s.add(tenant)
        s.flush()
        user = User(tenant_id=tenant.id, email="alice@acme.test", role=Role.EDITOR)
        s.add(user)
        s.flush()
        return tenant.id, user.id


class FakeEmbeddingProvider:
    """Deterministic pseudo-random unit vectors per text; records every text it embeds."""

    price_per_million_tokens = 1.0

    def __init__(self, model_name: str = "fake-embed-1", dim: int = EMBEDDING_DIM) -> None:
        self.model_name = model_name
        self.dim = dim
        self.embedded: list[str] = []
        self.fail = False

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(self.dim)
        return [float(x) for x in v / np.linalg.norm(v)]

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if self.fail:
            raise EmbeddingError("fake provider is down")
        self.embedded.extend(texts)
        return EmbeddingResult([self._vector(t) for t in texts], sum(len(t) for t in texts))

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise EmbeddingError("fake provider is down")
        return self._vector(text)


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


class FakeReranker:
    """Scores a document by the share of the query's words it contains (0..1)."""

    model_name = "fake-rerank-1"
    price_per_million_tokens = 2.0

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = False

    def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        if self.fail:
            raise RerankError("fake reranker is down")
        self.calls.append((query, list(documents)))
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scores = [
            len(words & set(re.findall(r"[a-z0-9]+", doc.lower()))) / max(len(words), 1)
            for doc in documents
        ]
        return RerankResult(scores, sum(len(d) for d in documents))


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


class InlineQueue:
    """Runs ingestion synchronously instead of sending it to Celery."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: LocalStorage,
        settings: Settings,
        embedder: FakeEmbeddingProvider,
    ) -> None:
        self.sessions, self.storage, self.settings = sessions, storage, settings
        self.embedder = embedder
        self.enqueued: list[uuid.UUID] = []
        self.fail = False

    def enqueue(self, document_id: uuid.UUID) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.enqueued.append(document_id)
        ingest_document(
            document_id,
            session_factory=self.sessions,
            storage=self.storage,
            settings=self.settings,
            embedder_factory=lambda: self.embedder,
        )


@pytest.fixture
def queue(
    sync_sessions: sessionmaker[Session],
    storage: LocalStorage,
    settings: Settings,
    embedder: FakeEmbeddingProvider,
) -> InlineQueue:
    return InlineQueue(sync_sessions, storage, settings, embedder)


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    """Override in a test module to change retrieval settings for the API client."""
    return RetrievalConfig(candidates=50, ef_search=100, rerank_candidates=30)


class Identity:
    """Lets a test switch which user the API sees (stand-in for auth until Phase 6)."""

    def __init__(self, user_id: uuid.UUID) -> None:
        self.user_id = user_id


@pytest.fixture
def identity(tenant_and_user: tuple[uuid.UUID, uuid.UUID]) -> Identity:
    return Identity(tenant_and_user[1])


class FakeAnswerGenerator:
    """Streams "According to the policy, <first paragraph of passage 1>" citing that paragraph."""

    model = "fake-claude"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ContextBlock]]] = []
        self.histories: list[tuple[list[HistoryMessage], str | None]] = []
        self.error: Exception | None = None
        self.hang = False  # stop after the first delta until cancelled (client disconnects)

    async def stream(
        self,
        question: str,
        blocks: Sequence[ContextBlock],
        history: Sequence[HistoryMessage] = (),
        summary: str | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        self.calls.append((question, list(blocks)))
        self.histories.append((list(history), summary))
        yield UsageUpdate(Usage(input_tokens=1200, cost_usd=0.0048), self.model)
        lead = "According to the policy, "
        yield TextDelta(lead)
        if self.hang:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        cited = blocks[0].segments[0].text
        yield TextDelta(cited)
        citation = CitationResolver(blocks).resolve(
            SimpleNamespace(
                document_index=0, start_block_index=0, end_block_index=1, cited_text=cited
            )
        )
        assert citation is not None
        yield CitationAdded(index=0, citation=citation, span_start=len(lead))
        yield Completed(
            text=lead + cited,
            spans=[Span(len(lead), len(lead) + len(cited), [0])],
            citations=[citation],
            stop_reason="end_turn",
            model=self.model,
            fallback_used=False,
            truncated=False,
            refusal_category=None,
            usage=Usage(
                input_tokens=1200, output_tokens=80, cache_read_tokens=640, cost_usd=0.0067
            ),
        )


class FakeConversationHelper:
    """Rewrites a follow-up as "<question> (about: <first earlier user message>)"."""

    model = "fake-haiku"

    def __init__(self) -> None:
        self.rewrites: list[tuple[str, list[HistoryMessage]]] = []
        self.summaries: list[tuple[str | None, list[HistoryMessage]]] = []
        self.error: LLMError | None = None

    async def rewrite(self, question: str, history: Sequence[HistoryMessage]) -> HelperResult:
        self.rewrites.append((question, list(history)))
        if self.error:
            raise self.error
        topic = next((m.text for m in history if m.role == "user"), "")
        return HelperResult(
            f"{question} (about: {topic})", Usage(input_tokens=90, cost_usd=0.00001)
        )

    async def summarize(
        self, previous: str | None, messages: Sequence[HistoryMessage]
    ) -> HelperResult:
        self.summaries.append((previous, list(messages)))
        return HelperResult(f"Summary of {len(messages)} messages", Usage(cost_usd=0.00002))


@pytest.fixture
def conversation_helper() -> FakeConversationHelper:
    return FakeConversationHelper()


@pytest.fixture
def answer_generator() -> FakeAnswerGenerator:
    return FakeAnswerGenerator()


@pytest.fixture
async def async_sessions(
    settings: Settings,
    sync_sessions: sessionmaker[Session],  # creates the schema
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.database_url)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def app(
    settings: Settings,
    storage: LocalStorage,
    queue: InlineQueue,
    identity: Identity,
    reranker: FakeReranker,
    retrieval_config: RetrievalConfig,
    answer_generator: FakeAnswerGenerator,
    conversation_helper: FakeConversationHelper,
    rate_limiter: MemoryRateLimiter,
    async_sessions: async_sessionmaker[AsyncSession],
) -> FastAPI:
    """The API with test doubles. Tests may change `app.dependency_overrides` at any time."""

    async def _db() -> AsyncIterator[AsyncSession]:
        async with async_sessions() as session:
            yield session

    async def _user() -> User:
        async with async_sessions() as session:
            user = await session.get(User, identity.user_id)
            assert user is not None
            return user

    retriever = HybridRetriever(
        async_sessions, lambda: queue.embedder, retrieval_config, reranker_factory=lambda: reranker
    )

    application = create_app(settings)
    application.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_db: _db,
            get_async_sessionmaker: lambda: async_sessions,
            get_storage: lambda: storage,
            get_ingestion_queue: lambda: queue,
            get_current_user: _user,
            get_retriever: lambda: retriever,
            get_answer_generator: lambda: answer_generator,
            get_conversation_helper: lambda: conversation_helper,
            get_rate_limiter: lambda: rate_limiter,
        }
    )
    return application


@pytest.fixture
def rate_limiter() -> MemoryRateLimiter:
    return MemoryRateLimiter()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
