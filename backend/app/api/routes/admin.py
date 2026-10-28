from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import distinct, func, select

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import IngestionReport
from app.core.errors import AppError
from app.db.models import Chunk, Document, DocumentPage, Role, User

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(user: CurrentUser) -> User:
    if user.role != Role.ADMIN:
        raise AppError("forbidden", "Administrator access is required.", 403)
    return user


AdminUser = Annotated[User, Depends(require_admin)]


@router.get("/ingestion-report")
async def ingestion_report(db: DbSession, user: AdminUser) -> IngestionReport:
    """Corpus-wide ingestion numbers for the admin's tenant: chunks, duplicates, embedding cost."""
    tenant = user.tenant_id
    by_status = {
        status: count
        for status, count in await db.execute(
            select(Document.status, func.count())
            .where(Document.tenant_id == tenant)
            .group_by(Document.status)
        )
    }
    pages_total = await db.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .join(Document, Document.id == DocumentPage.document_id)
        .where(Document.tenant_id == tenant)
    )
    chunks_total, distinct_texts, avg_tokens = (
        await db.execute(
            select(func.count(), func.count(distinct(Chunk.text_hash)), func.avg(Chunk.token_count))
            .select_from(Chunk)
            .where(Chunk.tenant_id == tenant)
        )
    ).one()
    by_model = {
        model or "none": count
        for model, count in await db.execute(
            select(Chunk.embedding_model, func.count())
            .where(Chunk.tenant_id == tenant)
            .group_by(Chunk.embedding_model)
        )
    }
    near_pending = await db.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.tenant_id == tenant, Document.near_duplicate_of_id.is_not(None))
    )

    # Per-document stats live in JSON metadata; summing in Python keeps this database-agnostic.
    metas = list(await db.scalars(select(Document.meta).where(Document.tenant_id == tenant)))
    embedding = [m.get("embedding", {}) for m in metas]
    return IngestionReport(
        documents_total=sum(by_status.values()),
        documents_by_status=by_status,
        pages_total=pages_total or 0,
        chunks_total=chunks_total,
        avg_chunk_tokens=round(float(avg_tokens), 1) if avg_tokens is not None else None,
        duplicate_chunks=chunks_total - distinct_texts,
        chunks_by_embedding_model=by_model,
        exact_duplicate_uploads=sum(m.get("duplicate_uploads", 0) for m in metas),
        near_duplicates_pending=near_pending or 0,
        embedding_tokens=sum(e.get("tokens", 0) for e in embedding),
        embedding_cost_usd=round(sum(e.get("cost_usd", 0.0) for e in embedding), 6),
        embedding_cache_hits=sum(e.get("cache_hits", 0) for e in embedding),
    )
