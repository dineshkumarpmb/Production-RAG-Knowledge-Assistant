import uuid
from functools import partial

from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.db.session import get_sync_sessionmaker
from app.embeddings import get_embedding_provider
from app.ingestion.pipeline import index_document, ingest_document, stale_document_ids
from app.storage.local import get_storage
from app.workers.celery_app import celery_app

_RETRY = {
    "autoretry_for": (OperationalError,),
    "retry_backoff": True,
    "retry_jitter": True,
    "max_retries": 3,
}


@celery_app.task(name="ingestion.ingest_document", soft_time_limit=1800, time_limit=1860, **_RETRY)
def ingest_document_task(document_id: str) -> None:
    settings = get_settings()
    ingest_document(
        uuid.UUID(document_id),
        session_factory=get_sync_sessionmaker(),
        storage=get_storage(),
        settings=settings,
        embedder_factory=partial(get_embedding_provider, settings),
    )


@celery_app.task(name="ingestion.index_document", soft_time_limit=1800, time_limit=1860, **_RETRY)
def index_document_task(document_id: str) -> None:
    """Re-chunk and re-embed a parsed document without parsing it again."""
    settings = get_settings()
    index_document(
        uuid.UUID(document_id),
        session_factory=get_sync_sessionmaker(),
        settings=settings,
        embedder_factory=partial(get_embedding_provider, settings),
    )


@celery_app.task(name="ingestion.reembed_stale")
def reembed_stale_task() -> int:
    """Queue re-embedding for documents embedded with a model other than the configured one.

    Run after changing EMBEDDING_MODEL:
        celery -A app.workers.celery_app:celery_app call ingestion.reembed_stale
    """
    settings = get_settings()
    model = get_embedding_provider(settings).model_name
    ids = stale_document_ids(get_sync_sessionmaker(), model)
    for document_id in ids:
        index_document_task.delay(str(document_id))
    return len(ids)
