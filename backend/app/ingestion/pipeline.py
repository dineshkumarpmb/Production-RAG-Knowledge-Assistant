"""Ingestion: stored file -> cleaned pages -> chunks -> embeddings, all in the database.

Status lifecycle: queued -> parsing -> embedding -> ready | failed.

Two stages, each idempotent (re-running replaces its output in one transaction):
- parse:  file -> pages + metadata + SimHash, with near-duplicate detection
- index:  pages -> chunks -> embeddings (cached). Runs on its own to re-embed after a model change.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import delete, or_, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Document, DocumentPage, DocumentStatus, FileType
from app.embeddings import EmbeddingError, EmbeddingProvider
from app.embeddings.cache import embed_with_cache
from app.ingestion.chunking import ChunkingConfig, chunk_pages, embedding_input
from app.ingestion.cleaning import HeadingRef, PageContent, clean_pages
from app.ingestion.dedup import find_near_duplicate, simhash, text_hash, to_signed64
from app.ingestion.language import detect_language
from app.ingestion.parsers import ParsedDocument, ParseError, parse_file
from app.storage.base import Storage

log = structlog.get_logger(__name__)

EmbedderFactory = Callable[[], EmbeddingProvider]

NO_TEXT_ERROR = (
    "No extractable text was found. The file may be a scanned document; OCR is not supported yet."
)
UNEXPECTED_ERROR = "An unexpected error occurred while processing the document."

_PLACEHOLDER_TITLES = {"untitled", "title", "document", "microsoft word"}
_MAX_LISTED_PAGES = 500
# Metadata written by the API rather than the parser; kept when a document is re-parsed.
_PRESERVED_META = ("duplicate_uploads", "near_duplicate_dismissed", "replaced_document")


@dataclass
class IngestionResult:
    pages: list[PageContent]
    low_text_pages: list[int]
    title: str
    author: str | None
    created_at: datetime | None
    language: str | None
    simhash: int
    metadata: dict[str, Any]


def chunking_config(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(
        target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
        contextual_header=settings.chunk_contextual_header,
    )


def _resolve_title(parsed: ParsedDocument, pages: list[PageContent], filename: str) -> str:
    """Prefer embedded metadata, then the first top-level heading, then the file name."""
    candidate = (parsed.title or "").strip()
    lowered = candidate.lower()
    if (
        candidate
        and lowered not in _PLACEHOLDER_TITLES
        and not lowered.startswith("microsoft word - ")
    ):
        return candidate[:500]
    for page in pages:
        for heading in page.headings:
            if heading.level == 1:
                return heading.text[:500]
    return Path(filename).stem[:500] or filename


def build_result(
    parsed: ParsedDocument, *, file_type: str, filename: str, settings: Settings
) -> IngestionResult:
    pages, removed_edges = clean_pages(parsed.pages)
    total_chars = sum(p.char_count for p in pages)
    if total_chars == 0:
        raise ParseError(NO_TEXT_ERROR)

    low_text = [p.number for p in pages if p.char_count < settings.low_text_page_chars]
    full_text = "\n\n".join(p.text for p in pages)
    metadata: dict[str, Any] = {
        **parsed.extra,
        "page_mode": parsed.page_mode,
        "char_count": total_chars,
        "heading_count": sum(len(p.headings) for p in pages),
        "removed_header_footer_blocks": removed_edges,
        "low_text_pages": low_text[:_MAX_LISTED_PAGES],
        # Mostly-empty PDFs are probably scans; OCR is a Phase 12 stretch goal.
        "needs_ocr": file_type == FileType.PDF and len(low_text) >= len(pages) / 2,
    }
    return IngestionResult(
        pages=pages,
        low_text_pages=low_text,
        title=_resolve_title(parsed, pages, filename),
        author=parsed.author[:500] if parsed.author else None,
        created_at=parsed.created_at,
        language=detect_language(full_text),
        simhash=simhash(full_text),
        metadata=metadata,
    )


def _mark_failed(
    session_factory: sessionmaker[Session], document_id: uuid.UUID, error: str
) -> None:
    with session_factory() as session, session.begin():
        doc = session.get(Document, document_id)
        if doc is not None:
            doc.status = DocumentStatus.FAILED
            doc.error = error


def _save_parse_result(
    session_factory: sessionmaker[Session],
    document_id: uuid.UUID,
    result: IngestionResult,
    settings: Settings,
) -> bool:
    low_text = set(result.low_text_pages)
    with session_factory() as session, session.begin():
        doc = session.get(Document, document_id, with_for_update=True)
        if doc is None:
            return False
        session.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))
        session.add_all(
            DocumentPage(
                document_id=document_id,
                page_number=page.number,
                text=page.text,
                char_count=page.char_count,
                is_low_text=page.number in low_text,
                headings=[asdict(h) for h in page.headings],
            )
            for page in result.pages
        )
        meta = {**result.metadata, **{k: doc.meta[k] for k in _PRESERVED_META if k in doc.meta}}

        dismissed = {uuid.UUID(i) for i in meta.get("near_duplicate_dismissed", [])}
        near = find_near_duplicate(
            session, doc, result.simhash, settings.near_duplicate_max_distance, exclude=dismissed
        )
        doc.near_duplicate_of_id = near.document_id if near else None
        if near:
            meta["near_duplicate"] = {
                "document_id": str(near.document_id),
                "title": near.title,
                "distance": near.distance,
            }

        doc.title = result.title
        doc.author = result.author
        doc.created_at_src = result.created_at
        doc.language = result.language
        doc.page_count = len(result.pages)
        doc.simhash = to_signed64(result.simhash)
        doc.meta = meta
        doc.status = DocumentStatus.EMBEDDING
        doc.error = None
    return True


def _parse(
    document_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session],
    storage: Storage,
    settings: Settings,
) -> bool:
    bound = log.bind(document_id=str(document_id))

    with session_factory() as session, session.begin():
        doc = session.get(Document, document_id, with_for_update=True)
        if doc is None:
            bound.warning("ingest.document_missing")
            return False
        if doc.status == DocumentStatus.DUPLICATE:
            return False
        doc.status = DocumentStatus.PARSING
        doc.error = None
        storage_uri, file_type, filename = doc.storage_uri, doc.file_type, doc.filename

    bound.info("ingest.started", file_type=file_type)
    started = time.perf_counter()
    try:
        with storage.local_path(storage_uri) as path:
            parsed = parse_file(path, file_type, settings)
        result = build_result(parsed, file_type=file_type, filename=filename, settings=settings)
    except ParseError as exc:
        bound.warning("ingest.parse_failed", error=str(exc))
        _mark_failed(session_factory, document_id, str(exc))
        return False
    except OperationalError:
        raise  # database trouble is transient: let the worker retry
    except Exception:
        bound.exception("ingest.crashed")
        _mark_failed(session_factory, document_id, UNEXPECTED_ERROR)
        return False

    if not _save_parse_result(session_factory, document_id, result, settings):
        bound.warning("ingest.document_deleted_during_processing")
        return False
    bound.info(
        "ingest.parsed",
        pages=len(result.pages),
        low_text_pages=len(result.low_text_pages),
        language=result.language,
        duration_ms=round((time.perf_counter() - started) * 1000),
    )
    return True


def _page_content(page: DocumentPage) -> PageContent:
    return PageContent(page.page_number, page.text, [HeadingRef(**h) for h in page.headings])


def index_document(
    document_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session],
    settings: Settings,
    embedder_factory: EmbedderFactory,
) -> None:
    """Chunk the document's saved pages, embed the chunks and replace its existing chunks."""
    bound = log.bind(document_id=str(document_id))
    with session_factory() as session, session.begin():
        doc = session.get(Document, document_id, with_for_update=True)
        if doc is None:
            return
        rows = session.scalars(
            select(DocumentPage)
            .where(DocumentPage.document_id == document_id)
            .order_by(DocumentPage.page_number)
        )
        pages = [_page_content(p) for p in rows]
        if not pages:
            bound.warning("index.no_pages")
            return
        doc.status = DocumentStatus.EMBEDDING
        title, tenant_id = doc.title, doc.tenant_id

    started = time.perf_counter()
    config = chunking_config(settings)
    chunks = chunk_pages(pages, config)
    inputs = [embedding_input(c, title, contextual_header=config.contextual_header) for c in chunks]
    try:
        provider = embedder_factory()
        vectors, stats = embed_with_cache(session_factory, provider, inputs)
    except EmbeddingError as exc:
        bound.warning("index.embedding_failed", error=str(exc))
        _mark_failed(session_factory, document_id, f"Embedding failed: {exc}")
        return
    except OperationalError:
        raise
    except Exception:
        bound.exception("index.crashed")
        _mark_failed(session_factory, document_id, UNEXPECTED_ERROR)
        return

    hashes = [text_hash(c.text) for c in chunks]
    with session_factory() as session, session.begin():
        doc = session.get(Document, document_id, with_for_update=True)
        if doc is None:
            bound.warning("index.document_deleted_during_processing")
            return
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        session.add_all(
            Chunk(
                document_id=document_id,
                tenant_id=tenant_id,
                chunk_index=c.index,
                text=c.text,
                text_hash=h,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                heading_path=list(c.heading_path),
                token_count=c.token_count,
                embedding=v,
                embedding_model=provider.model_name,
            )
            for c, h, v in zip(chunks, hashes, vectors, strict=True)
        )
        doc.meta = {
            **doc.meta,
            "chunking": {
                "target_tokens": config.target_tokens,
                "overlap_ratio": config.overlap_ratio,
                "contextual_header": config.contextual_header,
                "chunk_count": len(chunks),
                "avg_tokens": round(sum(c.token_count for c in chunks) / max(len(chunks), 1)),
                "duplicate_chunks": len(hashes) - len(set(hashes)),
            },
            "embedding": stats.as_dict(),
        }
        doc.status = DocumentStatus.READY
        doc.error = None
    bound.info(
        "index.completed",
        chunks=len(chunks),
        cache_hits=stats.cache_hits,
        embedded=stats.embedded,
        cost_usd=stats.cost_usd,
        duration_ms=round((time.perf_counter() - started) * 1000),
    )


def ingest_document(
    document_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session],
    storage: Storage,
    settings: Settings,
    embedder_factory: EmbedderFactory,
) -> None:
    if _parse(document_id, session_factory=session_factory, storage=storage, settings=settings):
        index_document(
            document_id,
            session_factory=session_factory,
            settings=settings,
            embedder_factory=embedder_factory,
        )


def stale_document_ids(session_factory: sessionmaker[Session], model: str) -> list[uuid.UUID]:
    """Ready documents with chunks embedded by a different model (or not embedded)."""
    with session_factory() as session:
        return list(
            session.scalars(
                select(Document.id)
                .where(Document.status == DocumentStatus.READY)
                .where(
                    select(Chunk.id)
                    .where(
                        Chunk.document_id == Document.id,
                        or_(Chunk.embedding_model.is_(None), Chunk.embedding_model != model),
                    )
                    .exists()
                )
            )
        )
