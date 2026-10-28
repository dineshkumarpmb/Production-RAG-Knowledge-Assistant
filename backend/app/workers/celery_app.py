from typing import Any

from celery import Celery
from celery.signals import setup_logging

from app.core.config import get_settings
from app.core.logging import configure_logging

INGESTION_QUEUE = "ingestion"


def create_celery() -> Celery:
    settings = get_settings()
    app = Celery("rag", broker=settings.redis_url, include=["app.workers.tasks"])
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        task_default_queue=INGESTION_QUEUE,
        task_ignore_result=True,
        # Acknowledge only after the task finishes, so a crashed worker's job is redelivered.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
    )
    return app


@setup_logging.connect
def _configure_worker_logging(**_: Any) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


celery_app = create_celery()
