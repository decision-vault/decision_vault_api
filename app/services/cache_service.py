from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as redis

from app.core.config import settings

_redis_client: redis.Redis | None = None
_memory_cache: dict[str, tuple[float, Any]] = {}


def _now() -> float:
    return time.time()


async def _get_redis() -> redis.Redis | None:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        await _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_client = None
        return None


def build_cache_key(*, feature: str, tenant_id: str, normalized_input: str) -> str:
    digest = hashlib.sha256(normalized_input.encode("utf-8")).hexdigest()
    return f"dv:{feature}:{tenant_id}:{digest}"


async def cache_get(key: str) -> Any | None:
    client = await _get_redis()
    if client:
        try:
            raw = await client.get(key)
            if raw:
                try:
                    return json.loads(raw)
                except Exception:
                    return None
        except Exception:
            # Graceful fallback to in-memory cache on Redis runtime failures.
            pass
    item = _memory_cache.get(key)
    if not item:
        return None
    expires_at, value = item
    if _now() >= expires_at:
        _memory_cache.pop(key, None)
        return None
    return value


async def cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    client = await _get_redis()
    if client:
        try:
            await client.set(key, json.dumps(value), ex=ttl_seconds)
            return
        except Exception:
            # Graceful fallback to in-memory cache on Redis runtime failures.
            pass
    _memory_cache[key] = (_now() + ttl_seconds, value)
