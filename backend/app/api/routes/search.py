import uuid
from datetime import datetime
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import AdminPrincipal, CurrentPrincipal, SettingsDep, rate_limit
from app.core.errors import AppError
from app.db.access import Principal
from app.db.models import FileType
from app.db.session import get_async_sessionmaker
from app.embeddings import get_embedding_provider
from app.rerank import get_reranker
from app.retrieval.context import ContextBlock
from app.retrieval.service import HybridRetriever, RetrievalConfig, VectorSearchUnavailable
from app.retrieval.types import Mode, RankedChunk, SearchFilters, SearchHit, SearchResult

router = APIRouter(tags=["search"])


def get_retriever(settings: SettingsDep) -> HybridRetriever:
    return HybridRetriever(
        get_async_sessionmaker(),
        partial(get_embedding_provider, settings),
        RetrievalConfig.from_settings(settings),
        reranker_factory=partial(get_reranker, settings),
    )


RetrieverDep = Annotated[HybridRetriever, Depends(get_retriever)]
_search_limit = Depends(rate_limit("search", "rate_limit_search_per_minute"))


class FiltersIn(BaseModel):
    document_ids: list[uuid.UUID] | None = Field(None, max_length=100)
    file_types: list[FileType] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None

    def to_filters(self) -> SearchFilters:
        return SearchFilters(
            document_ids=tuple(self.document_ids) if self.document_ids is not None else None,
            file_types=tuple(t.value for t in self.file_types) if self.file_types else None,
            date_from=self.date_from,
            date_to=self.date_to,
        )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Mode = Mode.HYBRID
    top_k: int = Field(10, ge=1, le=50)
    filters: FiltersIn = Field(default_factory=FiltersIn)
    collapse_duplicates: bool = Field(
        True, description="Return one chunk per identical text (e.g. repeated boilerplate)."
    )
    rerank: bool = Field(
        False, description="Rescore the top candidates with the cross-encoder (slower, better)."
    )


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str | None
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    heading_path: list[str]
    text: str
    score: float
    rerank_score: float | None
    fused_rank: int
    keyword_rank: int | None
    vector_rank: int | None
    keyword_score: float | None
    vector_similarity: float | None
    duplicates: int


class SearchResponse(BaseModel):
    query: str
    mode: Mode
    hits: list[SearchHitOut]
    not_found: bool = Field(
        description="No results, or (when reranked) the best score is below RERANK_MIN_SCORE"
    )
    degraded: list[str] = Field(description="Parts of retrieval that were skipped, if any")
    embedding_model: str | None
    reranker_model: str | None
    timings_ms: dict[str, float]


class DebugSearchRequest(SearchRequest):
    candidates: int = Field(50, ge=1, le=200)
    rrf_k: int = Field(60, ge=1, le=1000)
    keyword_weight: float = Field(1.0, ge=0, le=10)
    vector_weight: float = Field(1.0, ge=0, le=10)
    ef_search: int = Field(100, ge=10, le=1000)
    rerank_candidates: int = Field(30, ge=1, le=200)
    rerank_min_score: float | None = None


class RankedOut(BaseModel):
    rank: int
    chunk_id: uuid.UUID
    score: float


class DebugSearchResponse(SearchResponse):
    keyword: list[RankedOut]
    vector: list[RankedOut]
    fused: list[RankedOut]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=50, description="Chunks kept after reranking")
    token_budget: int | None = Field(None, ge=100, le=100_000)
    filters: FiltersIn = Field(default_factory=FiltersIn)


class ContextBlockOut(BaseModel):
    document_id: uuid.UUID
    document_title: str | None
    filename: str
    page_start: int
    page_end: int
    heading_path: list[str]
    chunk_ids: list[uuid.UUID]
    text: str
    score: float
    token_count: int


class RetrieveResponse(BaseModel):
    query: str
    blocks: list[ContextBlockOut]
    not_found: bool
    degraded: list[str]
    embedding_model: str | None
    reranker_model: str | None
    rerank_cost_usd: float
    timings_ms: dict[str, float]


def _hit_out(hit: SearchHit) -> SearchHitOut:
    return SearchHitOut.model_validate(hit, from_attributes=True)


def _block_out(block: ContextBlock) -> ContextBlockOut:
    return ContextBlockOut.model_validate(block, from_attributes=True)


def _ranked(items: list[RankedChunk]) -> list[RankedOut]:
    return [RankedOut(rank=i, chunk_id=r.chunk_id, score=r.score) for i, r in enumerate(items, 1)]


def _response(body: SearchRequest, result: SearchResult) -> SearchResponse:
    return SearchResponse(
        query=body.query,
        mode=body.mode,
        hits=[_hit_out(h) for h in result.hits],
        not_found=result.not_found,
        degraded=result.degraded,
        embedding_model=result.embedding_model,
        reranker_model=result.reranker_model,
        timings_ms=result.timings_ms,
    )


async def _search(
    retriever: HybridRetriever,
    principal: Principal,
    body: SearchRequest,
    config: RetrievalConfig | None = None,
) -> SearchResult:
    try:
        return await retriever.search(
            principal,
            body.query,
            mode=body.mode,
            top_k=body.top_k,
            filters=body.filters.to_filters(),
            config=config,
            collapse_duplicates=body.collapse_duplicates,
            rerank=body.rerank,
        )
    except VectorSearchUnavailable as exc:
        raise AppError("vector_unavailable", f"Vector search is unavailable: {exc}", 503) from exc


@router.post("/search", dependencies=[_search_limit])
async def search(
    body: SearchRequest, principal: CurrentPrincipal, retriever: RetrieverDep
) -> SearchResponse:
    """Hybrid (keyword + vector) search over the chunks of documents the user can read."""
    return _response(body, await _search(retriever, principal, body))


@router.post("/retrieve", dependencies=[_search_limit])
async def retrieve(
    body: RetrieveRequest,
    principal: CurrentPrincipal,
    retriever: RetrieverDep,
    settings: SettingsDep,
) -> RetrieveResponse:
    """The context an answer would be grounded in: hybrid search, reranking, then adjacent
    chunks merged into passages within a token budget."""
    result = await retriever.retrieve(
        principal,
        body.query,
        top_k=body.top_k or settings.rerank_top_k,
        token_budget=body.token_budget or settings.context_token_budget,
        filters=body.filters.to_filters(),
    )
    search = result.search
    return RetrieveResponse(
        query=body.query,
        blocks=[_block_out(b) for b in result.blocks],
        not_found=search.not_found,
        degraded=search.degraded,
        embedding_model=search.embedding_model,
        reranker_model=search.reranker_model,
        rerank_cost_usd=search.rerank_cost_usd,
        timings_ms=search.timings_ms,
    )


@router.post("/admin/search/debug", tags=["admin"])
async def search_debug(
    body: DebugSearchRequest, principal: AdminPrincipal, retriever: RetrieverDep
) -> DebugSearchResponse:
    """Keyword, vector and fused rankings side by side, with tunable parameters. With
    `rerank`, each hit also shows its rank before reranking (`fused_rank`)."""
    config = RetrievalConfig(
        candidates=body.candidates,
        rrf_k=body.rrf_k,
        keyword_weight=body.keyword_weight,
        vector_weight=body.vector_weight,
        ef_search=body.ef_search,
        rerank_candidates=body.rerank_candidates,
        rerank_min_score=body.rerank_min_score,
    )
    result = await _search(retriever, principal, body, config)
    return DebugSearchResponse(
        **_response(body, result).model_dump(),
        keyword=_ranked(result.keyword),
        vector=_ranked(result.vector),
        fused=[
            RankedOut(rank=i, chunk_id=f.chunk_id, score=f.score)
            for i, f in enumerate(result.fused, 1)
        ],
    )
