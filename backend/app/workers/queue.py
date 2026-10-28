import uuid
from typing import Protocol


class IngestionQueue(Protocol):
    def enqueue(self, document_id: uuid.UUID) -> None: ...


class CeleryIngestionQueue:
    def enqueue(self, document_id: uuid.UUID) -> None:
        from app.workers.tasks import ingest_document_task

        ingest_document_task.delay(str(document_id))


def get_ingestion_queue() -> IngestionQueue:
    return CeleryIngestionQueue()
