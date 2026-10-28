import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.api.deps import CurrentUser, DbSession, QueueDep, SettingsDep, StorageDep
from app.api.schemas import (
    ChunkList,
    ChunkOut,
    DocumentList,
    DocumentOut,
    NearDuplicateResolution,
    PageList,
    PageOut,
    UploadResponse,
)
from app.core.errors import AppError, NotFoundError
from app.db.access import Principal, can_read
from app.db.models import Chunk, Document, DocumentPage, DocumentStatus, Role, User, Visibility
from app.ingestion.validation import (
    MIME_TYPES,
    check_content,
    file_type_from_name,
    sanitize_filename,
)
from app.workers.queue import IngestionQueue

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_CHUNK = 1024 * 1024
_MULTIPART_OVERHEAD = 64 * 1024
_QUEUE_FAILED_ERROR = "Could not queue the document for processing. Use reprocess to retry."
_IN_PROGRESS = (DocumentStatus.QUEUED, DocumentStatus.PARSING, DocumentStatus.EMBEDDING)


async def _get_visible(db: AsyncSession, user: User, document_id: uuid.UUID) -> Document:
    doc = (
        await db.execute(
            select(Document).where(Document.id == document_id, can_read(Principal.of(user)))
        )
    ).scalar_one_or_none()
    if doc is None:
        raise NotFoundError("Document")  # same answer whether it is missing or forbidden
    return doc


async def _spool_upload(file: UploadFile, dest: Path, max_bytes: int) -> tuple[int, str]:
    """Copy the upload to `dest`, enforcing the size limit and hashing as we go."""
    digest = hashlib.sha256()
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(_READ_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise AppError(
                    "file_too_large",
                    f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
                    413,
                )
            digest.update(chunk)
            out.write(chunk)
    return size, digest.hexdigest()


async def _duplicate_response(
    db: AsyncSession, user: User, sha256: str, response: Response
) -> UploadResponse:
    existing = (
        await db.execute(
            select(Document).where(Document.tenant_id == user.tenant_id, Document.sha256 == sha256)
        )
    ).scalar_one()
    if existing.owner_id != user.id and existing.visibility != Visibility.TENANT:
        # Don't reveal another user's private document; the uploader already has the content.
        raise AppError(
            "duplicate_document",
            "This file already exists in your organisation's library. Ask its owner to share it.",
            409,
        )
    existing.meta = {
        **existing.meta,
        "duplicate_uploads": existing.meta.get("duplicate_uploads", 0) + 1,
    }
    await db.commit()
    await db.refresh(existing)
    response.status_code = status.HTTP_200_OK
    return UploadResponse(document=DocumentOut.model_validate(existing), duplicate=True)


def _require_owner(doc: Document, user: User, action: str) -> None:
    if doc.owner_id != user.id and user.role != Role.ADMIN:
        raise AppError("forbidden", f"Only the owner can {action} this document.", 403)


async def _delete_document(db: AsyncSession, doc_id: uuid.UUID) -> None:
    """Delete a document and everything derived from it (the caller commits).

    Explicit deletes rather than relying on FK cascades, which SQLite does not enforce by default.
    """
    await db.execute(
        update(Document)
        .where(Document.near_duplicate_of_id == doc_id)
        .values(near_duplicate_of_id=None)
    )
    await db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
    await db.execute(delete(DocumentPage).where(DocumentPage.document_id == doc_id))
    await db.execute(delete(Document).where(Document.id == doc_id))


async def _enqueue(db: AsyncSession, queue: IngestionQueue, doc: Document) -> None:
    try:
        await run_in_threadpool(queue.enqueue, doc.id)
    except Exception as exc:
        log.exception("ingest.enqueue_failed", document_id=str(doc.id))
        doc.status = DocumentStatus.FAILED
        doc.error = _QUEUE_FAILED_ERROR
        await db.commit()
        raise AppError("queue_unavailable", _QUEUE_FAILED_ERROR, 503) from exc


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": UploadResponse, "description": "Exact duplicate of an upload"}},
)
async def upload_document(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or Markdown file")],
    db: DbSession,
    user: CurrentUser,
    storage: StorageDep,
    queue: QueueDep,
    settings: SettingsDep,
) -> UploadResponse:
    """Upload a document. Parsing happens in the background; poll `GET /documents/{id}`."""
    declared = int(request.headers.get("content-length") or 0)
    if declared > settings.max_upload_bytes + _MULTIPART_OVERHEAD:
        raise AppError("file_too_large", "File exceeds the upload limit.", 413)

    filename = sanitize_filename(file.filename or "")
    file_type = file_type_from_name(filename)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "upload"
        size, sha256 = await _spool_upload(file, tmp_path, settings.max_upload_bytes)
        if size == 0:
            raise AppError("empty_file", "The uploaded file is empty.", 422)
        await run_in_threadpool(check_content, tmp_path, file_type, settings)

        exists = await db.scalar(
            select(Document.id).where(
                Document.tenant_id == user.tenant_id, Document.sha256 == sha256
            )
        )
        if exists:
            return await _duplicate_response(db, user, sha256, response)

        doc_id = uuid.uuid4()
        key = f"{user.tenant_id}/{doc_id}/original.{file_type.value}"
        storage_uri = await run_in_threadpool(storage.put_file, key, tmp_path)

    doc = Document(
        id=doc_id,
        tenant_id=user.tenant_id,
        owner_id=user.id,
        filename=filename,
        file_type=file_type,
        mime_type=MIME_TYPES[file_type],
        size_bytes=size,
        sha256=sha256,
        storage_uri=storage_uri,
        status=DocumentStatus.QUEUED,
        meta={},
    )
    db.add(doc)
    try:
        await db.commit()
    except IntegrityError:  # a concurrent upload of the same file won the race
        await db.rollback()
        await run_in_threadpool(storage.delete, storage_uri)
        return await _duplicate_response(db, user, sha256, response)

    log.info("document.uploaded", document_id=str(doc.id), file_type=file_type, size=size)
    await _enqueue(db, queue, doc)
    await db.refresh(doc)
    return UploadResponse(document=DocumentOut.model_validate(doc), duplicate=False)


@router.get("")
async def list_documents(
    db: DbSession,
    user: CurrentUser,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    where = [can_read(Principal.of(user))]
    if status_filter is not None:
        where.append(Document.status == status_filter)
    total = await db.scalar(select(func.count()).select_from(Document).where(*where)) or 0
    rows = await db.scalars(
        select(Document)
        .where(*where)
        .order_by(Document.created_at.desc(), Document.id)
        .limit(limit)
        .offset(offset)
    )
    return DocumentList(
        items=[DocumentOut.model_validate(d) for d in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}")
async def get_document(document_id: uuid.UUID, db: DbSession, user: CurrentUser) -> DocumentOut:
    return DocumentOut.model_validate(await _get_visible(db, user, document_id))


@router.get("/{document_id}/pages")
async def list_pages(
    document_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PageList:
    """Cleaned page text with heading positions (useful for checking parser output)."""
    doc = await _get_visible(db, user, document_id)
    total = (
        await db.scalar(
            select(func.count()).select_from(DocumentPage).where(DocumentPage.document_id == doc.id)
        )
        or 0
    )
    pages = await db.scalars(
        select(DocumentPage)
        .where(DocumentPage.document_id == doc.id)
        .order_by(DocumentPage.page_number)
        .limit(limit)
        .offset(offset)
    )
    return PageList(
        document_id=doc.id,
        page_mode=doc.meta.get("page_mode"),
        items=[PageOut.model_validate(p) for p in pages],
        total=total,
    )


@router.post("/{document_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_document(
    document_id: uuid.UUID, db: DbSession, user: CurrentUser, queue: QueueDep
) -> DocumentOut:
    """Re-run ingestion (e.g. after a failure or a parser upgrade). Replaces pages and chunks."""
    doc = await _get_visible(db, user, document_id)
    _require_owner(doc, user, "reprocess")
    if doc.status in _IN_PROGRESS:
        raise AppError("already_processing", "The document is already being processed.", 409)

    doc.status = DocumentStatus.QUEUED
    doc.error = None
    await db.commit()
    await _enqueue(db, queue, doc)
    await db.refresh(doc)
    return DocumentOut.model_validate(doc)


@router.get("/{document_id}/chunks")
async def list_chunks(
    document_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkList:
    """Chunks with their page ranges and heading paths (embeddings omitted)."""
    doc = await _get_visible(db, user, document_id)
    total = (
        await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id))
        or 0
    )
    chunks = await db.scalars(
        select(Chunk)
        .options(defer(Chunk.embedding))
        .where(Chunk.document_id == doc.id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    return ChunkList(
        document_id=doc.id, items=[ChunkOut.model_validate(c) for c in chunks], total=total
    )


@router.post("/{document_id}/near-duplicate")
async def resolve_near_duplicate(
    document_id: uuid.UUID,
    body: NearDuplicateResolution,
    db: DbSession,
    user: CurrentUser,
    storage: StorageDep,
) -> DocumentOut:
    """Decide what to do when an upload is nearly identical to an existing document."""
    doc = await _get_visible(db, user, document_id)
    _require_owner(doc, user, "resolve duplicates for")
    if doc.near_duplicate_of_id is None:
        raise AppError(
            "no_near_duplicate", "This document is not flagged as a near-duplicate.", 409
        )

    meta = {k: v for k, v in doc.meta.items() if k != "near_duplicate"}
    old = await db.get(Document, doc.near_duplicate_of_id)
    old_uri: str | None = None
    if body.action == "keep_both" or old is None:
        dismissed = {*meta.get("near_duplicate_dismissed", []), str(doc.near_duplicate_of_id)}
        meta["near_duplicate_dismissed"] = sorted(dismissed)
    else:
        _require_owner(old, user, "replace")
        if old.status in _IN_PROGRESS:
            raise AppError("already_processing", "The older document is still processing.", 409)
        meta["replaced_document"] = {
            "document_id": str(old.id),
            "title": old.title,
            "version": old.version,
        }
        doc.version = old.version + 1
        old_uri = old.storage_uri
        await _delete_document(db, old.id)

    doc.near_duplicate_of_id = None
    doc.meta = meta
    await db.commit()
    if old_uri:
        await run_in_threadpool(storage.delete, old_uri)
        log.info("document.replaced", document_id=str(doc.id), replaced=meta["replaced_document"])
    await db.refresh(doc)
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, db: DbSession, user: CurrentUser, storage: StorageDep
) -> None:
    """Delete a document, its pages, chunks and stored file."""
    doc = await _get_visible(db, user, document_id)
    _require_owner(doc, user, "delete")
    storage_uri = doc.storage_uri
    await _delete_document(db, doc.id)
    await db.commit()
    await run_in_threadpool(storage.delete, storage_uri)
    log.info("document.deleted", document_id=str(document_id))
