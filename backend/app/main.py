from fastapi import FastAPI

from app.api.routes import admin, chat, documents, health, search
from app.core.config import Settings, get_settings
from app.core.errors import RequestIdMiddleware, register_exception_handlers
from app.core.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(admin.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    return app


app = create_app()
