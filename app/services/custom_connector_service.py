from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe

from jose import jwt

from app.core.config import settings
from app.db.mongo import get_db
from app.services.license_service import evaluate_license_status, get_current_license


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def generate_api_key(tenant_id: str) -> str:
    db = get_db()
    key = token_urlsafe(32)
    key_hash = _hash_key(key)
    await db.custom_connector_keys.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "key_hash": key_hash,
                "created_at": _utcnow(),
                "rotated_at": _utcnow(),
            }
        },
        upsert=True,
    )
    return key


async def rotate_api_key(tenant_id: str) -> str:
    return await generate_api_key(tenant_id)


async def verify_api_key(tenant_id: str, api_key: str) -> bool:
    db = get_db()
    doc = await db.custom_connector_keys.find_one({"tenant_id": tenant_id})
    if not doc:
        return False
    return hmac.compare_digest(doc["key_hash"], _hash_key(api_key))


def verify_hmac_signature(raw_body: bytes, signature: str) -> bool:
    if not settings.custom_connector_hmac_secret:
        return False
    expected = hmac.new(
        settings.custom_connector_hmac_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def license_allows_capture(tenant_id: str) -> bool:
    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        return False
    status = evaluate_license_status(license_doc)["status"]
    return status not in {"expired", "grace", "suspended"}


async def create_oauth_client(tenant_id: str, name: str | None = None) -> dict:
    db = get_db()
    client_id = str(uuid.uuid4())
    client_secret = token_urlsafe(32)
    secret_hash = _hash_key(client_secret)
    doc = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret_hash": secret_hash,
        "name": name,
        "created_at": _utcnow(),
        "rotated_at": _utcnow(),
    }
    await db.custom_connector_oauth_clients.update_one(
        {"tenant_id": tenant_id},
        {"$set": doc},
        upsert=True,
    )
    return {"client_id": client_id, "client_secret": client_secret, "name": name}


async def rotate_oauth_client_secret(tenant_id: str) -> dict:
    return await create_oauth_client(tenant_id)


async def verify_oauth_client(client_id: str, client_secret: str) -> str | None:
    db = get_db()
    doc = await db.custom_connector_oauth_clients.find_one({"client_id": client_id})
    if not doc:
        return None
    if not hmac.compare_digest(doc["client_secret_hash"], _hash_key(client_secret)):
        return None
    return doc["tenant_id"]


def create_oauth_access_token(tenant_id: str, client_id: str) -> tuple[str, int]:
    expires = _utcnow() + timedelta(minutes=settings.custom_connector_oauth_token_minutes)
    payload = {
        "sub": client_id,
        "tenant_id": tenant_id,
        "type": "custom_oauth",
        "exp": int(expires.timestamp()),
        "iat": int(_utcnow().timestamp()),
        "iss": settings.jwt_issuer,
        "aud": settings.custom_connector_oauth_audience,
        "scope": "ingest",
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return token, settings.custom_connector_oauth_token_minutes * 60


def verify_oauth_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.custom_connector_oauth_audience,
            issuer=settings.jwt_issuer,
        )
        if payload.get("type") != "custom_oauth":
            return None
        return payload
    except Exception:
        return None


async def record_delivery(
    tenant_id: str,
    external_id: str,
    status: str,
    request_id: str,
    error: str | None = None,
    metadata: dict | None = None,
) -> None:
    db = get_db()
    await db.custom_connector_deliveries.insert_one(
        {
            "tenant_id": tenant_id,
            "external_id": external_id,
            "status": status,
            "error": error,
            "metadata": metadata or {},
            "request_id": request_id,
            "created_at": _utcnow(),
        }
    )


async def enqueue_retry(payload: dict, error: str, attempt: int = 1) -> None:
    db = get_db()
    delay = compute_retry_delay_seconds(attempt)
    next_attempt = _utcnow() + timedelta(seconds=delay)
    await db.custom_connector_retry_queue.insert_one(
        {
            "tenant_id": payload["tenant_id"],
            "external_id": payload["external_id"],
            "payload": payload,
            "attempt": attempt,
            "next_attempt_at": next_attempt,
            "last_error": error,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
        }
    )


def compute_retry_delay_seconds(attempt: int) -> int:
    base = settings.custom_connector_retry_base_seconds
    max_delay = settings.custom_connector_retry_max_seconds
    return int(min(base * (2 ** max(attempt - 1, 0)), max_delay))


async def get_due_retries(limit: int = 50) -> list[dict]:
    db = get_db()
    cursor = (
        db.custom_connector_retry_queue.find({"next_attempt_at": {"$lte": _utcnow()}})
        .sort("next_attempt_at", 1)
        .limit(limit)
    )
    return [doc async for doc in cursor]


async def update_retry_attempt(retry_id, attempt: int, error: str | None, next_attempt_at: datetime | None) -> None:
    db = get_db()
    updates = {"attempt": attempt, "last_error": error, "updated_at": _utcnow()}
    if next_attempt_at:
        updates["next_attempt_at"] = next_attempt_at
    await db.custom_connector_retry_queue.update_one({"_id": retry_id}, {"$set": updates})


async def mark_retry_complete(retry_id) -> None:
    db = get_db()
    await db.custom_connector_retry_queue.delete_one({"_id": retry_id})


async def connector_health(tenant_id: str) -> dict:
    db = get_db()
    last_delivery = await db.custom_connector_deliveries.find_one(
        {"tenant_id": tenant_id}, sort=[("created_at", -1)]
    )
    last_success = await db.custom_connector_deliveries.find_one(
        {"tenant_id": tenant_id, "status": "success"}, sort=[("created_at", -1)]
    )
    last_error = await db.custom_connector_deliveries.find_one(
        {"tenant_id": tenant_id, "status": "error"}, sort=[("created_at", -1)]
    )
    retry_count = await db.custom_connector_retry_queue.count_documents({"tenant_id": tenant_id})
    since = _utcnow() - timedelta(hours=24)
    total_24h = await db.custom_connector_deliveries.count_documents(
        {"tenant_id": tenant_id, "created_at": {"$gte": since}}
    )
    success_24h = await db.custom_connector_deliveries.count_documents(
        {"tenant_id": tenant_id, "created_at": {"$gte": since}, "status": "success"}
    )
    return {
        "last_delivery_at": last_delivery.get("created_at") if last_delivery else None,
        "last_success_at": last_success.get("created_at") if last_success else None,
        "last_error_at": last_error.get("created_at") if last_error else None,
        "retry_queue_depth": retry_count,
        "deliveries_last_24h": total_24h,
        "success_rate_last_24h": (success_24h / total_24h) if total_24h else None,
    }
