from __future__ import annotations

import asyncpg

from app.core.config import settings

_pool: asyncpg.Pool | None = None


async def get_pg_pool() -> asyncpg.Pool:
    global _pool
    if _pool:
        return _pool
    if not settings.postgres_dsn:
        raise RuntimeError("Postgres DSN not configured")
    _pool = await asyncpg.create_pool(dsn=settings.postgres_dsn, min_size=1, max_size=5)
    return _pool
