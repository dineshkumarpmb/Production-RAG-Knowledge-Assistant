from typing import Annotated

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.models import Role, Tenant, User
from app.db.session import get_db
from app.storage.base import Storage
from app.storage.local import get_storage
from app.workers.queue import IngestionQueue, get_ingestion_queue

SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
StorageDep = Annotated[Storage, Depends(get_storage)]
QueueDep = Annotated[IngestionQueue, Depends(get_ingestion_queue)]


async def _find_user(db: AsyncSession, email: str) -> User | None:
    return (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()


async def get_current_user(db: DbSession, settings: SettingsDep) -> User:
    """Placeholder identity: every request acts as one dev user.

    Replaced by JWT authentication in Phase 6. Refuses to run in production.
    """
    if settings.is_prod:
        raise AppError("unauthenticated", "Authentication is not configured.", 401)
    if user := await _find_user(db, settings.dev_user_email):
        return user

    tenant = (
        await db.execute(select(Tenant).where(Tenant.name == settings.dev_tenant_name))
    ).scalar_one_or_none() or Tenant(name=settings.dev_tenant_name)
    db.add(tenant)
    await db.flush()
    user = User(tenant_id=tenant.id, email=settings.dev_user_email, role=Role.ADMIN)
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:  # another request created it first
        await db.rollback()
        if existing := await _find_user(db, settings.dev_user_email):
            return existing
        raise
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
