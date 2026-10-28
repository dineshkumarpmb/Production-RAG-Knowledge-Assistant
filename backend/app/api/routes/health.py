import asyncio

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text

from app.api.deps import DbSession, SettingsDep

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])

_CHECK_TIMEOUT_S = 2.0


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(db: DbSession, settings: SettingsDep) -> JSONResponse:
    """Readiness: the database and Redis are reachable."""
    checks: dict[str, str] = {}

    try:
        await asyncio.wait_for(db.execute(text("SELECT 1")), _CHECK_TIMEOUT_S)
        checks["database"] = "ok"
    except Exception:
        log.warning("readyz.database_unavailable", exc_info=True)
        checks["database"] = "unavailable"

    redis = Redis.from_url(settings.redis_url, socket_connect_timeout=_CHECK_TIMEOUT_S)
    try:
        await asyncio.wait_for(redis.ping(), _CHECK_TIMEOUT_S)
        checks["redis"] = "ok"
    except Exception:
        log.warning("readyz.redis_unavailable", exc_info=True)
        checks["redis"] = "unavailable"
    finally:
        await redis.aclose()

    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if ready else "unavailable", "checks": checks},
        status_code=200 if ready else 503,
    )
