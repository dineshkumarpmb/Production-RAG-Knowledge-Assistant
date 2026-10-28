from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, auth, chat, documents, health, search
from app.core.config import Settings, get_settings
from app.core.errors import RequestIdMiddleware, register_exception_handlers
from app.core.logging import configure_logging
from app.core.security import check_secret
from app.core.web import SecurityHeadersMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    check_secret(settings)  # refuse to start in production with the development JWT secret

    app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # bearer tokens, not cookies
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(admin.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    return app


app = create_app()
