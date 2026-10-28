"""Per-user rate limits: fixed one-minute windows counted in Redis.

If Redis is unreachable the request is allowed and a warning is logged: a limiter outage should
not take the whole API down with it.
"""

import time
from collections import Counter
from functools import lru_cache
from typing import Protocol

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings

log = structlog.get_logger(__name__)

WINDOW_SECONDS = 60


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int) -> int | None:
        """Count one request; return seconds to wait if over `limit`, else None."""
        ...


def _window() -> tuple[int, int]:
    now = int(time.time())
    start = now - now % WINDOW_SECONDS
    return start, start + WINDOW_SECONDS - now


class RedisRateLimiter:
    def __init__(self, url: str) -> None:
        self._redis = Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)

    async def hit(self, key: str, limit: int) -> int | None:
        start, retry_after = _window()
        bucket = f"ratelimit:{key}:{start}"
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                count, _ = await pipe.incr(bucket).expire(bucket, WINDOW_SECONDS + 5).execute()
        except RedisError as exc:
            log.warning("ratelimit.unavailable", error=str(exc))
            return None
        return retry_after if count > limit else None


class MemoryRateLimiter:
    """In-process limiter for tests and single-process use."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    async def hit(self, key: str, limit: int) -> int | None:
        start, retry_after = _window()
        bucket = f"{key}:{start}"
        self._counts[bucket] += 1
        return retry_after if self._counts[bucket] > limit else None


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RedisRateLimiter(get_settings().redis_url)
