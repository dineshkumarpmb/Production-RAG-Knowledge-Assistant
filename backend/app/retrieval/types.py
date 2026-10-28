import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Mode(StrEnum):
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


class Retriever(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"


@dataclass(frozen=True)
class SearchFilters:
    document_ids: tuple[uuid.UUID, ...] | None = None
    file_types: tuple[str, ...] | None = None
    # Applied to the document's own date when known, otherwise its upload date.
    date_from: datetime | None = None
    date_to: datetime | None = None


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: uuid.UUID
    score: float  # ts_rank_cd for keyword, cosine similarity for vector


@dataclass
class FusedChunk:
    chunk_id: uuid.UUID
    score: float
    ranks: dict[Retriever, int] = field(default_factory=dict)
    scores: dict[Retriever, float] = field(default_factory=dict)


@dataclass
class SearchHit:
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
    keyword_rank: int | None
    vector_rank: int | None
    keyword_score: float | None
    vector_similarity: float | None
    duplicates: int  # other chunks with identical text that were collapsed into this one
    fused_rank: int  # position after fusion (and collapsing), before any reranking
    rerank_score: float | None = None  # cross-encoder relevance, 0..1 for local models


@dataclass
class SearchResult:
    hits: list[SearchHit]
    keyword: list[RankedChunk]
    vector: list[RankedChunk]
    fused: list[FusedChunk]
    degraded: list[str]  # e.g. ["vector_unavailable"] when the embedding provider failed
    embedding_model: str | None
    timings_ms: dict[str, float]
    reranker_model: str | None = None
    rerank_tokens: int = 0
    rerank_cost_usd: float = 0.0
    not_found: bool = False  # no hits, or the best reranked score is below the threshold
