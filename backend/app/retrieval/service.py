"""Hybrid retrieval: keyword and vector search run concurrently and are fused with RRF;
then, optionally, a cross-encoder reranks the top candidates.

Failures degrade instead of failing the request, and say so in `degraded`: no query
embedding means keyword-only results, and no reranker means the fused order is kept.
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.access import Principal, RlsContext, set_rls_async
from app.db.models import Chunk, Document
from app.embeddings import EmbeddingError, EmbeddingProvider
from app.ingestion.chunking import with_source_header
from app.rerank import Reranker, RerankError
from app.retrieval.context import ContextBlock, assemble_context
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.queries import keyword_search, vector_search
from app.retrieval.types import (
    FusedChunk,
    Mode,
    RankedChunk,
    Retriever,
    SearchFilters,
    SearchHit,
    SearchResult,
)

log = structlog.get_logger(__name__)

VECTOR_UNAVAILABLE = "vector_unavailable"
RERANK_UNAVAILABLE = "rerank_unavailable"


class VectorSearchUnavailable(Exception):
    """Raised for vector-only searches when no query embedding can be produced."""


@dataclass(frozen=True)
class RetrievalConfig:
    candidates: int = 50
    rrf_k: int = 60
    keyword_weight: float = 1.0
    vector_weight: float = 1.0
    ef_search: int = 100
    rerank_candidates: int = 30
    rerank_min_score: float | None = None  # applies only when results were reranked

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrievalConfig":
        return cls(
            candidates=settings.retrieval_candidates,
            rrf_k=settings.retrieval_rrf_k,
            keyword_weight=settings.retrieval_keyword_weight,
            vector_weight=settings.retrieval_vector_weight,
            ef_search=settings.hnsw_ef_search,
            rerank_candidates=settings.rerank_candidates,
            rerank_min_score=settings.rerank_min_score,
        )


@dataclass
class RetrievalResult:
    """What answer generation consumes: assembled context plus how it was found."""

    blocks: list[ContextBlock]
    search: SearchResult


async def _timed[T](timings: dict[str, float], name: str, work: Awaitable[T]) -> T:
    started = time.perf_counter()
    try:
        return await work
    finally:
        timings[name] = round((time.perf_counter() - started) * 1000, 2)


class HybridRetriever:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder_factory: Callable[[], EmbeddingProvider],
        config: RetrievalConfig,
        reranker_factory: Callable[[], Reranker | None] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._embedder_factory = embedder_factory
        self._reranker_factory = reranker_factory
        self.config = config

    async def search(
        self,
        principal: Principal,
        query: str,
        *,
        mode: Mode = Mode.HYBRID,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        config: RetrievalConfig | None = None,
        collapse_duplicates: bool = True,
        rerank: bool = False,
    ) -> SearchResult:
        cfg = config or self.config
        filters = filters or SearchFilters()
        # Every session is scoped to this user for Postgres row-level security as well.
        rls = RlsContext.for_principal(principal)
        timings: dict[str, float] = {}
        degraded: list[str] = []
        embedding_model: list[str] = []
        started = time.perf_counter()

        async def run_keyword() -> list[RankedChunk]:
            async with self._sessions() as session:
                await set_rls_async(session, rls)
                return await _timed(
                    timings,
                    "keyword",
                    keyword_search(
                        session, principal, query, limit=cfg.candidates, filters=filters
                    ),
                )

        async def run_vector() -> list[RankedChunk]:
            try:
                provider = await asyncio.to_thread(self._embedder_factory)
                vector = await _timed(
                    timings, "embed_query", asyncio.to_thread(provider.embed_query, query)
                )
            except EmbeddingError as exc:
                log.warning("search.vector_unavailable", error=str(exc))
                if mode == Mode.VECTOR:
                    raise VectorSearchUnavailable(str(exc)) from exc
                degraded.append(VECTOR_UNAVAILABLE)
                return []
            embedding_model.append(provider.model_name)
            async with self._sessions() as session:
                await set_rls_async(session, rls)
                return await _timed(
                    timings,
                    "vector",
                    vector_search(
                        session,
                        principal,
                        vector,
                        model=provider.model_name,
                        limit=cfg.candidates,
                        ef_search=max(cfg.ef_search, cfg.candidates),
                        filters=filters,
                    ),
                )

        async def nothing() -> list[RankedChunk]:
            return []

        keyword, vector = await asyncio.gather(
            run_keyword() if mode != Mode.VECTOR else nothing(),
            run_vector() if mode != Mode.KEYWORD else nothing(),
        )

        rankings = {Retriever.KEYWORD: keyword, Retriever.VECTOR: vector}
        fused = reciprocal_rank_fusion(
            {r: ranking for r, ranking in rankings.items() if ranking},
            k=cfg.rrf_k,
            weights={Retriever.KEYWORD: cfg.keyword_weight, Retriever.VECTOR: cfg.vector_weight},
        )
        pool = max(top_k, cfg.rerank_candidates) if rerank else top_k
        hits = await _timed(
            timings, "hydrate", self._hydrate(fused, pool, rls, collapse=collapse_duplicates)
        )
        reranked = await self._rerank(query, hits, timings, degraded) if rerank and hits else None
        hits = hits[:top_k]
        not_found = not hits or (
            reranked is not None
            and cfg.rerank_min_score is not None
            and (hits[0].rerank_score or 0.0) < cfg.rerank_min_score
        )
        timings["total"] = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "search.completed",
            mode=mode,
            keyword=len(keyword),
            vector=len(vector),
            hits=len(hits),
            reranked=reranked is not None,
            not_found=not_found,
            degraded=degraded,
            total_ms=timings["total"],
        )
        return SearchResult(
            hits=hits,
            keyword=keyword,
            vector=vector,
            fused=fused,
            degraded=degraded,
            embedding_model=embedding_model[0] if embedding_model else None,
            timings_ms=timings,
            reranker_model=reranked.model if reranked else None,
            rerank_tokens=reranked.tokens if reranked else 0,
            rerank_cost_usd=reranked.cost_usd if reranked else 0.0,
            not_found=not_found,
        )

    async def retrieve(
        self,
        principal: Principal,
        query: str,
        *,
        top_k: int,
        token_budget: int,
        filters: SearchFilters | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """The answer-generation path: hybrid search, rerank, then assemble the context."""
        result = await self.search(
            principal, query, top_k=top_k, filters=filters, config=config, rerank=True
        )
        blocks = (
            [] if result.not_found else assemble_context(result.hits, token_budget=token_budget)
        )
        return RetrievalResult(blocks=blocks, search=result)

    async def _rerank(
        self, query: str, hits: list[SearchHit], timings: dict[str, float], degraded: list[str]
    ) -> "_Reranked | None":
        """Rescore `hits` in place with the cross-encoder and sort them by that score."""
        if self._reranker_factory is None:
            return None
        texts = [with_source_header(h.document_title, h.heading_path, h.text) for h in hits]
        try:
            reranker = await asyncio.to_thread(self._reranker_factory)
            if reranker is None:  # reranking switched off in settings
                return None
            result = await _timed(
                timings, "rerank", asyncio.to_thread(reranker.rerank, query, texts)
            )
            if len(result.scores) != len(hits):
                raise RerankError(f"{len(result.scores)} scores for {len(hits)} chunks")
        except RerankError as exc:
            log.warning("search.rerank_unavailable", error=str(exc))
            degraded.append(RERANK_UNAVAILABLE)
            return None
        for hit, score in zip(hits, result.scores, strict=True):
            hit.rerank_score = score
        hits.sort(key=lambda h: -(h.rerank_score or 0.0))  # stable: ties keep fused order
        cost = result.tokens * reranker.price_per_million_tokens / 1_000_000
        return _Reranked(reranker.model_name, result.tokens, round(cost, 6))

    async def _hydrate(
        self, fused: list[FusedChunk], top_k: int, rls: RlsContext, *, collapse: bool
    ) -> list[SearchHit]:
        """Load chunk and document details, collapsing chunks with identical text."""
        if not fused:
            return []
        ids = [f.chunk_id for f in fused]
        async with self._sessions() as session:
            await set_rls_async(session, rls)
            rows = await session.execute(
                select(
                    Chunk.id,
                    Chunk.document_id,
                    Chunk.chunk_index,
                    Chunk.page_start,
                    Chunk.page_end,
                    Chunk.heading_path,
                    Chunk.text,
                    Chunk.text_hash,
                    Document.title,
                    Document.filename,
                )
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(ids))
            )
            details: dict[uuid.UUID, Any] = {row.id: row for row in rows}

        hits: list[SearchHit] = []
        by_text: dict[str, SearchHit] = {}
        for f in fused:
            row = details.get(f.chunk_id)
            if row is None:
                continue  # deleted between search and hydration
            if collapse and (kept := by_text.get(row.text_hash)) is not None:
                kept.duplicates += 1
                continue
            if len(hits) == top_k:
                if not collapse:
                    break
                continue  # keep counting duplicates of the hits already chosen
            hit = SearchHit(
                chunk_id=row.id,
                document_id=row.document_id,
                document_title=row.title,
                filename=row.filename,
                chunk_index=row.chunk_index,
                page_start=row.page_start,
                page_end=row.page_end,
                heading_path=list(row.heading_path),
                text=row.text,
                score=f.score,
                keyword_rank=f.ranks.get(Retriever.KEYWORD),
                vector_rank=f.ranks.get(Retriever.VECTOR),
                keyword_score=f.scores.get(Retriever.KEYWORD),
                vector_similarity=f.scores.get(Retriever.VECTOR),
                duplicates=0,
                fused_rank=len(hits) + 1,
            )
            hits.append(hit)
            by_text[row.text_hash] = hit
        return hits


@dataclass(frozen=True)
class _Reranked:
    model: str
    tokens: int
    cost_usd: float
