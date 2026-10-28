"""Documents: upload, read, share, process and delete.

Permissions (see app.db.access):
- read     owner, everyone in the tenant (visibility "tenant"), or a read/write grant to the
           user or one of their groups
- write    owner, admin, or a write grant: reprocess, resolve near-duplicates
- manage   owner or admin: share, change visibility, delete
Anything a user cannot read answers 404, never 403, so IDs reveal nothing.
"""

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.api.deps import (
    CurrentPrincipal,
    DbSession,
    EditorPrincipal,
    QueueDep,
    SettingsDep,
    StorageDep,
    rate_limit,
)
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
from app.core.audit import audit
from app.core.errors import AppError, NotFoundError
from app.db.access import Principal, RlsContext, can_read, can_write, set_rls_async
from app.db.models import (
    Chunk,
    Document,
    DocumentAcl,
    DocumentPage,
    DocumentStatus,
    Group,
    Permission,
    PrincipalType,
    User,
    Visibility,
)
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


class ShareRequest(BaseModel):
    principal_type: PrincipalType
    principal_id: uuid.UUID
    permission: Permission = Permission.READ


class AccessEntry(BaseModel):
    id: uuid.UUID
    principal_type: str
    principal_id: uuid.UUID
    name: str  # user email or group name
    permission: str


class AccessOut(BaseModel):
    document_id: uuid.UUID
    owner_id: uuid.UUID
    visibility: str
    entries: list[AccessEntry]


class DocumentUpdate(BaseModel):
    visibility: Literal["private", "tenant"]


# --- permission helpers --------------------------------------------------------------------


async def _get_visible(db: AsyncSession, principal: Principal, document_id: uuid.UUID) -> Document:
    doc = (
        await db.execute(select(Document).where(Document.id == document_id, can_read(principal)))
    ).scalar_one_or_none()
    if doc is None:
        raise NotFoundError("Document")  # same answer whether it is missing or forbidden
    return doc


def _require_manage(doc: Document, principal: Principal, action: str) -> None:
    if doc.owner_id != principal.user_id and not principal.is_admin:
        raise AppError("forbidden", f"Only the owner can {action} this document.", 403)


async def _require_write(
    db: AsyncSession, doc: Document, principal: Principal, action: str
) -> None:
    if doc.owner_id == principal.user_id or principal.is_admin:
        return
    if not await db.scalar(select(Document.id).where(Document.id == doc.id, can_write(principal))):
        raise AppError("forbidden", f"You do not have write access to {action} this document.", 403)


# --- upload --------------------------------------------------------------------------------


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


async def _exists_in_tenant(db: AsyncSession, principal: Principal, sha256: str) -> bool:
    """Whether anyone in the tenant uploaded this exact file. Needs a tenant-wide look past
    row-level security; only a yes/no leaves this function."""
    await set_rls_async(db, RlsContext.tenant_wide(principal.tenant_id))
    try:
        found = await db.scalar(
            select(Document.id).where(
                Document.tenant_id == principal.tenant_id, Document.sha256 == sha256
            )
        )
    finally:
        await set_rls_async(db, RlsContext.for_principal(principal))
    return found is not None


async def _duplicate_response(
    db: AsyncSession, principal: Principal, sha256: str, response: Response
) -> UploadResponse:
    existing = (
        await db.execute(
            select(Document).where(
                Document.tenant_id == principal.tenant_id,
                Document.sha256 == sha256,
                can_read(principal),
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        # Someone else's document the uploader cannot read: say it exists, reveal nothing more.
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


async def _delete_document(db: AsyncSession, doc_id: uuid.UUID) -> None:
    """Delete a document and everything derived from it (the caller commits).

    Explicit deletes rather than relying on FK cascades, which SQLite does not enforce by default.
    """
    await db.execute(
        update(Document)
        .where(Document.near_duplicate_of_id == doc_id)
        .values(near_duplicate_of_id=None)
    )
    await db.execute(delete(DocumentAcl).where(DocumentAcl.document_id == doc_id))
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
    dependencies=[Depends(rate_limit("upload", "rate_limit_upload_per_minute"))],
)
async def upload_document(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or Markdown file")],
    db: DbSession,
    principal: EditorPrincipal,
    storage: StorageDep,
    queue: QueueDep,
    settings: SettingsDep,
) -> UploadResponse:
    """Upload a document (editors and admins). Parsing happens in the background; poll
    `GET /documents/{id}`. New documents are private to the uploader until shared."""
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

        if await _exists_in_tenant(db, principal, sha256):
            return await _duplicate_response(db, principal, sha256, response)

        doc_id = uuid.uuid4()
        key = f"{principal.tenant_id}/{doc_id}/original.{file_type.value}"
        storage_uri = await run_in_threadpool(storage.put_file, key, tmp_path)

    doc = Document(
        id=doc_id,
        tenant_id=principal.tenant_id,
        owner_id=principal.user_id,
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
    audit(
        db,
        "document.upload",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="document",
        target_id=doc_id,
        request=request,
        filename=filename,
        size=size,
    )
    try:
        await db.commit()
    except IntegrityError:  # a concurrent upload of the same file won the race
        await db.rollback()
        await run_in_threadpool(storage.delete, storage_uri)
        return await _duplicate_response(db, principal, sha256, response)

    log.info("document.uploaded", document_id=str(doc.id), file_type=file_type, size=size)
    await _enqueue(db, queue, doc)
    await db.refresh(doc)
    return UploadResponse(document=DocumentOut.model_validate(doc), duplicate=False)


# --- read ----------------------------------------------------------------------------------


@router.get("")
async def list_documents(
    db: DbSession,
    principal: CurrentPrincipal,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    where = [can_read(principal)]
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
async def get_document(
    document_id: uuid.UUID, db: DbSession, principal: CurrentPrincipal
) -> DocumentOut:
    return DocumentOut.model_validate(await _get_visible(db, principal, document_id))


@router.get("/{document_id}/file")
async def download_document(
    document_id: uuid.UUID,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
    storage: StorageDep,
) -> FileResponse:
    """The original file, e.g. for opening a cited page in the viewer."""
    doc = await _get_visible(db, principal, document_id)
    audit(
        db,
        "document.download",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="document",
        target_id=doc.id,
        request=request,
    )
    await db.commit()
    with storage.local_path(doc.storage_uri) as path:  # local storage: the file stays put
        return FileResponse(path, media_type=doc.mime_type, filename=doc.filename)


@router.get("/{document_id}/pages")
async def list_pages(
    document_id: uuid.UUID,
    db: DbSession,
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PageList:
    """Cleaned page text with heading positions (useful for checking parser output)."""
    doc = await _get_visible(db, principal, document_id)
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


@router.get("/{document_id}/chunks")
async def list_chunks(
    document_id: uuid.UUID,
    db: DbSession,
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkList:
    """Chunks with their page ranges and heading paths (embeddings omitted)."""
    doc = await _get_visible(db, principal, document_id)
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


# --- sharing -------------------------------------------------------------------------------


async def _access(db: AsyncSession, doc: Document) -> AccessOut:
    entries = list(await db.scalars(select(DocumentAcl).where(DocumentAcl.document_id == doc.id)))
    user_ids = [e.principal_id for e in entries if e.principal_type == PrincipalType.USER]
    group_ids = [e.principal_id for e in entries if e.principal_type == PrincipalType.GROUP]
    names = {
        **dict((await db.execute(select(User.id, User.email).where(User.id.in_(user_ids)))).all()),
        **dict(
            (await db.execute(select(Group.id, Group.name).where(Group.id.in_(group_ids)))).all()
        ),
    }
    return AccessOut(
        document_id=doc.id,
        owner_id=doc.owner_id,
        visibility=doc.visibility,
        entries=[
            AccessEntry(
                id=e.id,
                principal_type=e.principal_type,
                principal_id=e.principal_id,
                name=names.get(e.principal_id, ""),
                permission=e.permission,
            )
            for e in entries
        ],
    )


@router.get("/{document_id}/access")
async def get_access(
    document_id: uuid.UUID, db: DbSession, principal: CurrentPrincipal
) -> AccessOut:
    """Who can read or write this document (owner and admins)."""
    doc = await _get_visible(db, principal, document_id)
    _require_manage(doc, principal, "view sharing for")
    return await _access(db, doc)


@router.post("/{document_id}/access")
async def share_document(
    document_id: uuid.UUID,
    body: ShareRequest,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
) -> AccessOut:
    """Grant a user or group of the same organisation read or write access (or change it)."""
    doc = await _get_visible(db, principal, document_id)
    _require_manage(doc, principal, "share")
    target: User | Group | None
    if body.principal_type == PrincipalType.USER:
        target = await db.get(User, body.principal_id)
    else:
        target = await db.get(Group, body.principal_id)
    if target is None or target.tenant_id != principal.tenant_id:
        raise NotFoundError("User" if body.principal_type == PrincipalType.USER else "Group")

    entry = (
        await db.execute(
            select(DocumentAcl).where(
                DocumentAcl.document_id == doc.id,
                DocumentAcl.principal_type == body.principal_type,
                DocumentAcl.principal_id == body.principal_id,
            )
        )
    ).scalar_one_or_none()
    if entry is None:
        db.add(
            DocumentAcl(
                document_id=doc.id,
                principal_type=body.principal_type,
                principal_id=body.principal_id,
                permission=body.permission,
                granted_by=principal.user_id,
            )
        )
    else:
        entry.permission = body.permission
    audit(
        db,
        "document.share",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="document",
        target_id=doc.id,
        request=request,
        principal_type=body.principal_type,
        principal_id=str(body.principal_id),
        permission=body.permission,
    )
    await db.commit()
    return await _access(db, doc)


@router.delete("/{document_id}/access/{entry_id}")
async def unshare_document(
    document_id: uuid.UUID,
    entry_id: uuid.UUID,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
) -> AccessOut:
    doc = await _get_visible(db, principal, document_id)
    _require_manage(doc, principal, "change sharing for")
    entry = await db.get(DocumentAcl, entry_id)
    if entry is None or entry.document_id != doc.id:
        raise NotFoundError("Access entry")
    await db.delete(entry)
    audit(
        db,
        "document.unshare",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="document",
        target_id=doc.id,
        request=request,
        principal_type=entry.principal_type,
        principal_id=str(entry.principal_id),
    )
    await db.commit()
    return await _access(db, doc)


@router.patch("/{document_id}")
async def update_document(
    document_id: uuid.UUID,
    body: DocumentUpdate,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
) -> DocumentOut:
    """Share with the whole organisation (`tenant`) or make private again (`private`)."""
    doc = await _get_visible(db, principal, document_id)
    _require_manage(doc, principal, "change the visibility of")
    doc.visibility = Visibility(body.visibility)
    audit(
        db,
        "document.visibility",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="document",
        target_id=doc.id,
        request=request,
        visibility=body.visibility,
    )
    await db.commit()
    await db.refresh(doc)
    return DocumentOut.model_validate(doc)


# --- processing ----------------------------------------------------------------------------


@router.post("/{document_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_document(
    document_id: uuid.UUID, db: DbSession, principal: CurrentPrincipal, queue: QueueDep
) -> DocumentOut:
    """Re-run ingestion (e.g. after a failure or a parser upgrade). Replaces pages and chunks."""
    doc = await _get_visible(db, principal, document_id)
    await _require_write(db, doc, principal, "reprocess")
    if doc.status in _IN_PROGRESS:
        raise AppError("already_processing", "The document is already being processed.", 409)

    doc.status = DocumentStatus.QUEUED
    doc.error = None
    await db.commit()
    await _enqueue(db, queue, doc)
    await db.refresh(doc)
    return DocumentOut.model_validate(doc)


@router.post("/{document_id}/near-duplicate")
async def resolve_near_duplicate(
    document_id: uuid.UUID,
    body: NearDuplicateResolution,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
    storage: StorageDep,
) -> DocumentOut:
    """Decide what to do when an upload is nearly identical to an existing document."""
    doc = await _get_visible(db, principal, document_id)
    await _require_write(db, doc, principal, "resolve duplicates for")
    if doc.near_duplicate_of_id is None:
        raise AppError(
            "no_near_duplicate", "This document is not flagged as a near-duplicate.", 409
        )

    meta = {k: v for k, v in doc.meta.items() if k != "near_duplicate"}
    old = (
        await db.execute(
            select(Document).where(Document.id == doc.near_duplicate_of_id, can_read(principal))
        )
    ).scalar_one_or_none()
    old_uri: str | None = None
    if body.action == "keep_both" or old is None:
        dismissed = {*meta.get("near_duplicate_dismissed", []), str(doc.near_duplicate_of_id)}
        meta["near_duplicate_dismissed"] = sorted(dismissed)
    else:
        _require_manage(old, principal, "replace")
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
        audit(
            db,
            "document.replace",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            target_type="document",
            target_id=old.id,
            request=request,
            replaced_by=str(doc.id),
        )

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
    document_id: uuid.UUID,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
    storage: StorageDep,
) -> None:
    """Delete a document, its pages, chunks and stored file (owner or admin)."""
    doc = await _get_visible(db, principal, document_id)
    _require_manage(doc, principal, "delete")
    storage_uri = doc.storage_uri
    await _delete_document(db, doc.id)
    audit(
        db,
        "document.delete",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="document",
        target_id=document_id,
        request=request,
        filename=doc.filename,
    )
    await db.commit()
    await run_in_threadpool(storage.delete, storage_uri)
    log.info("document.deleted", document_id=str(document_id))
