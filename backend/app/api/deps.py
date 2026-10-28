from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.ratelimit import RateLimiter, get_rate_limiter
from app.core.security import InvalidTokenError, decode_access_token
from app.db.access import Principal, RlsContext, set_rls_async
from app.db.models import Role, User, UserGroup
from app.db.session import get_db
from app.storage.base import Storage
from app.storage.local import get_storage
from app.workers.queue import IngestionQueue, get_ingestion_queue

SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
StorageDep = Annotated[Storage, Depends(get_storage)]
QueueDep = Annotated[IngestionQueue, Depends(get_ingestion_queue)]
LimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]

_bearer = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _unauthenticated(message: str = "Sign in required.") -> AppError:
    return AppError("unauthenticated", message, 401, headers=_CHALLENGE)


async def get_current_user(
    db: DbSession, settings: SettingsDep, token: Annotated[str | None, Depends(_bearer)]
) -> User:
    """The signed-in user, from a bearer access token. Role and active status come from the
    database, not the token, so changes apply immediately."""
    if not token:
        raise _unauthenticated()
    try:
        claims = decode_access_token(settings, token)
    except InvalidTokenError as exc:
        raise _unauthenticated("The access token is invalid or has expired.") from exc
    user = await db.get(User, claims.user_id)
    if user is None or not user.is_active or user.tenant_id != claims.tenant_id:
        raise _unauthenticated("The access token is invalid or has expired.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_principal(db: DbSession, user: CurrentUser) -> Principal:
    """The user with their groups. Also scopes the request's database session to what this user
    may read, for Postgres row-level security."""
    group_ids = tuple(
        await db.scalars(select(UserGroup.group_id).where(UserGroup.user_id == user.id))
    )
    principal = Principal.of(user, group_ids)
    await set_rls_async(db, RlsContext.for_principal(principal))
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def require_role(*roles: Role) -> Callable[[Principal], Awaitable[Principal]]:
    async def check(principal: CurrentPrincipal) -> Principal:
        if principal.role not in roles:
            raise AppError("forbidden", "You do not have permission to do this.", 403)
        return principal

    return check


AdminPrincipal = Annotated[Principal, Depends(require_role(Role.ADMIN))]
EditorPrincipal = Annotated[Principal, Depends(require_role(Role.ADMIN, Role.EDITOR))]


def rate_limit(bucket: str, setting: str) -> Callable[..., Awaitable[None]]:
    """Dependency limiting each user to `getattr(settings, setting)` requests per minute."""

    async def check(user: CurrentUser, limiter: LimiterDep, settings: SettingsDep) -> None:
        retry_after = await limiter.hit(f"{bucket}:{user.id}", getattr(settings, setting))
        if retry_after is not None:
            raise AppError(
                "rate_limited",
                "Too many requests. Please wait a moment and try again.",
                429,
                headers={"Retry-After": str(retry_after)},
            )

    return check
