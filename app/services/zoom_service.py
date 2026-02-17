from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import httpx

from app.core.config import settings
from app.db.mongo import get_db
from app.services.crypto_service import decrypt_token_with_key, encrypt_token_with_key
from app.services.license_service import evaluate_license_status, get_current_license


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _token_expired(expires_at: datetime | None) -> bool:
    if not expires_at:
        return True
    return _utcnow() >= expires_at


async def store_installation(tenant_id: str, account_id: str, tokens: dict) -> None:
    db = get_db()
    await db.zoom_installations.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "account_id": account_id,
                "access_token": encrypt_token_with_key(
                    tokens["access_token"], settings.zoom_token_encryption_key
                ),
                "refresh_token": encrypt_token_with_key(
                    tokens["refresh_token"], settings.zoom_token_encryption_key
                ),
                "expires_at": _utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
                "installed_at": _utcnow(),
                "meeting_ids": [],
                "chat_channel_ids": [],
                "allow_chat": True,
            }
        },
        upsert=True,
    )


async def get_installation(tenant_id: str) -> dict | None:
    db = get_db()
    return await db.zoom_installations.find_one({"tenant_id": tenant_id})


async def get_installation_by_account(account_id: str) -> dict | None:
    db = get_db()
    return await db.zoom_installations.find_one({"account_id": account_id})


async def list_chat_channels(installation: dict) -> list[dict]:
    cache_age = None
    if installation.get("channels_cache_at"):
        cache_age = (_utcnow() - installation["channels_cache_at"]).total_seconds()
    if installation.get("channels_cache") and cache_age is not None:
        if cache_age < settings.zoom_channel_cache_seconds:
            return installation["channels_cache"]

    token = await get_access_token(installation)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.zoom.us/v2/chat/channels", headers=headers)
    response.raise_for_status()
    data = response.json()
    result = [{"id": c["id"], "name": c.get("name")} for c in data.get("channels", [])]
    db = get_db()
    await db.zoom_installations.update_one(
        {"_id": installation["_id"]},
        {"$set": {"channels_cache": result, "channels_cache_at": _utcnow()}},
    )
    return result


async def list_meetings(installation: dict) -> list[dict]:
    token = await get_access_token(installation)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.zoom.us/v2/users/me/meetings", headers=headers)
    response.raise_for_status()
    data = response.json()
    return [{"id": m["id"], "topic": m.get("topic")} for m in data.get("meetings", [])]


async def record_webhook_event(event_id: str) -> bool:
    db = get_db()
    existing = await db.zoom_webhook_events.find_one({"event_id": event_id})
    if existing:
        return False
    await db.zoom_webhook_events.insert_one({"event_id": event_id, "created_at": _utcnow()})
    return True


async def mark_webhook_received(tenant_id: str) -> None:
    db = get_db()
    await db.zoom_installations.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {"last_webhook_at": _utcnow()},
            "$inc": {"webhook_count": 1},
        },
    )


async def ensure_tenant_account_binding(tenant_id: str, account_id: str) -> None:
    db = get_db()
    existing_account = await db.zoom_installations.find_one({"account_id": account_id})
    if existing_account and existing_account.get("tenant_id") != tenant_id:
        raise ValueError("Zoom account already linked to another tenant")

    existing_tenant = await db.zoom_installations.find_one({"tenant_id": tenant_id})
    if existing_tenant and existing_tenant.get("account_id") != account_id:
        raise ValueError("Tenant already linked to another Zoom account")


async def set_scopes(
    tenant_id: str, meeting_ids: list[str], chat_channel_ids: list[str], allow_chat: bool
) -> None:
    db = get_db()
    await db.zoom_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"meeting_ids": meeting_ids, "chat_channel_ids": chat_channel_ids, "allow_chat": allow_chat}},
    )


async def license_allows_capture(tenant_id: str) -> bool:
    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        return False
    status = evaluate_license_status(license_doc)["status"]
    return status not in {"expired", "grace", "suspended"}


async def exchange_code_for_tokens(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://zoom.us/oauth/token",
            params={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.zoom_redirect_uri,
            },
            auth=(settings.zoom_client_id, settings.zoom_client_secret),
        )
    response.raise_for_status()
    return response.json()


async def refresh_tokens(installation: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://zoom.us/oauth/token",
            params={"grant_type": "refresh_token", "refresh_token": decrypt_token_with_key(
                installation["refresh_token"], settings.zoom_token_encryption_key
            )},
            auth=(settings.zoom_client_id, settings.zoom_client_secret),
        )
    response.raise_for_status()
    return response.json()


async def get_access_token(installation: dict) -> str:
    if _token_expired(installation.get("expires_at")):
        tokens = await refresh_tokens(installation)
        await store_installation(installation["tenant_id"], installation["account_id"], tokens)
        return tokens["access_token"]
    return decrypt_token_with_key(installation["access_token"], settings.zoom_token_encryption_key)


def verify_zoom_signature(raw_body: bytes, timestamp: str, signature: str) -> bool:
    if not timestamp or not signature:
        return False
    base = f"v0:{timestamp}:{raw_body.decode('utf-8')}"
    computed = (
        "v0="
        + hmac.new(
            settings.zoom_webhook_secret.encode("utf-8"),
            base.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )
    return hmac.compare_digest(computed, signature)


def parse_webhook_payload(raw_body: bytes) -> dict:
    return json.loads(raw_body.decode("utf-8"))


def is_decision_marker(text: str) -> bool:
    lowered = text.lower()
    keywords = ["decision:", "decision made", "final decision", "resolved"]
    return any(keyword in lowered for keyword in keywords)


def build_decision_record(source: str, tenant_id: str, payload: dict) -> dict:
    return {
        "tenant_id": tenant_id,
        "source": source,
        "thread_id": payload.get("thread_id"),
        "meeting_id": payload.get("meeting_id"),
        "channel_id": payload.get("channel_id"),
        "message_id": payload.get("message_id"),
        "text": payload.get("text"),
        "created_at": _utcnow(),
    }
