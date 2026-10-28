from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import URL, create_engine, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def sync_url(url: str) -> URL:
    """The API uses async drivers; the worker and Alembic use the sync equivalent."""
    parsed = make_url(url)
    if parsed.drivername == "sqlite+aiosqlite":
        return parsed.set(drivername="sqlite")
    return parsed  # psycopg 3 serves both sync and async from the same URL


@lru_cache
def get_async_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_async_engine(), expire_on_commit=False)


@lru_cache
def get_sync_sessionmaker() -> sessionmaker[Session]:
    engine = create_engine(sync_url(get_settings().database_url), pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_async_sessionmaker()() as session:
        yield session
