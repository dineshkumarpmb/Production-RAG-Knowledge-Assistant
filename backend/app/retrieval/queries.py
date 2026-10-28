"""Keyword and vector search queries (PostgreSQL). Both apply the same access filter."""

import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, cast, func, literal_column, select
from sqlalchemy.dialects.postgresql import TSQUERY, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.access import Principal, can_read
from app.db.models import Chunk, Document, DocumentStatus
from app.retrieval.types import RankedChunk, SearchFilters

TS_CONFIG: ColumnElement[Any] = literal_column(
    "'english'::regconfig"
)  # must match the chunks.tsv definition
TSV = literal_column("chunks.tsv", type_=TSVECTOR)
_RANK_NORMALISATION = 32  # ts_rank_cd option: rank / (rank + 1), keeps scores in [0, 1)
# Quotes, a leading -word, or an upper-case OR (lower-case "or" is normal in questions).
_SEARCH_SYNTAX = re.compile(r'"|(?:^|\s)-\w|\sOR\s')


def access_conditions(principal: Principal, filters: SearchFilters) -> list[ColumnElement[bool]]:
    """Conditions on the joined Document row: who may read it, plus the request's filters."""
    conditions = [can_read(principal), Document.status != DocumentStatus.FAILED]
    if filters.document_ids is not None:
        conditions.append(Document.id.in_(filters.document_ids))
    if filters.file_types is not None:
        conditions.append(Document.file_type.in_(filters.file_types))
    doc_date = func.coalesce(Document.created_at_src, Document.created_at)
    if filters.date_from is not None:
        conditions.append(doc_date >= filters.date_from)
    if filters.date_to is not None:
        conditions.append(doc_date <= filters.date_to)
    return conditions


def uses_search_syntax(query: str) -> bool:
    return bool(_SEARCH_SYNTAX.search(query))


def build_tsquery(query: str) -> ColumnElement[Any]:
    """Match ANY word of a plain query; honour explicit syntax when the user writes it.

    Questions are the normal input, and requiring every word ("how many days of PTO do
    employees get") would miss most relevant chunks. Matching any word and letting the rank
    favour chunks that match more of them behaves like BM25. Queries with "quoted phrases",
    OR or -excluded words go through websearch_to_tsquery, which applies that syntax strictly.
    """
    if uses_search_syntax(query):
        return func.websearch_to_tsquery(TS_CONFIG, query)
    words = func.unnest(func.to_tsvector(TS_CONFIG, query)).table_valued("lexeme")
    any_word = (
        select(func.string_agg(func.quote_literal(words.c.lexeme), " | "))
        .select_from(words)
        .scalar_subquery()
    )
    return cast(func.coalesce(any_word, ""), TSQUERY)


async def keyword_search(
    session: AsyncSession,
    principal: Principal,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
) -> list[RankedChunk]:
    tsquery = build_tsquery(query)
    rank = func.ts_rank_cd(TSV, tsquery, _RANK_NORMALISATION).label("score")
    stmt = (
        select(Chunk.id, rank)
        .join(Document, Document.id == Chunk.document_id)
        .where(TSV.op("@@")(tsquery), *access_conditions(principal, filters))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return [RankedChunk(chunk_id, float(score)) for chunk_id, score in await session.execute(stmt)]


async def vector_search(
    session: AsyncSession,
    principal: Principal,
    query_vector: Sequence[float],
    *,
    model: str,
    limit: int,
    ef_search: int,
    filters: SearchFilters,
) -> list[RankedChunk]:
    # Settings are transaction-local. Iterative scans (pgvector >= 0.8) keep scanning the HNSW
    # index until enough rows pass the access filter, instead of filtering a fixed-size result.
    await session.execute(select(func.set_config("hnsw.ef_search", str(ef_search), True)))
    await session.execute(select(func.set_config("hnsw.iterative_scan", "relaxed_order", True)))

    distance = Chunk.embedding.cosine_distance(list(query_vector)).label("distance")
    stmt = (
        select(Chunk.id, distance)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.embedding.is_not(None),
            # Vectors from different models are not comparable.
            Chunk.embedding_model == model,
            *access_conditions(principal, filters),
        )
        .order_by(distance)
        .limit(limit)
    )
    rows = sorted(await session.execute(stmt), key=lambda r: r.distance)  # relaxed_order
    return [RankedChunk(chunk_id, 1.0 - float(dist)) for chunk_id, dist in rows]
