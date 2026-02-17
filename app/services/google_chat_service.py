from __future__ import annotations

import json
from datetime import datetime, timezone
import hashlib
import hmac

import httpx
from google.oauth2 import service_account

from app.core.config import settings
from app.db.mongo import get_db
from app.services.license_service import evaluate_license_status, get_current_license


SCOPES = ["https://www.googleapis.com/auth/chat.messages.readonly"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _credentials():
    if not settings.google_chat_service_account_json:
        raise ValueError("Missing Google Chat service account config")
    info = json.loads(settings.google_chat_service_account_json)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    if settings.google_chat_delegated_user:
        creds = creds.with_subject(settings.google_chat_delegated_user)
    return creds


async def get_access_token() -> str:
    creds = _credentials()
    from google.auth.transport.requests import Request as GoogleRequest

    creds.refresh(GoogleRequest())
    return creds.token


async def store_installation(tenant_id: str, domain: str) -> None:
    db = get_db()
    await db.google_chat_installations.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "domain": domain,
                "installed_at": _utcnow(),
                "spaces": [],
                "allow_direct_messages": False,
            }
        },
        upsert=True,
    )


async def ensure_tenant_domain_binding(tenant_id: str, domain: str) -> None:
    db = get_db()
    existing_domain = await db.google_chat_installations.find_one({"domain": domain})
    if existing_domain and existing_domain.get("tenant_id") != tenant_id:
        raise ValueError("Google Workspace domain already linked to another tenant")

    existing_tenant = await db.google_chat_installations.find_one({"tenant_id": tenant_id})
    if existing_tenant and existing_tenant.get("domain") != domain:
        raise ValueError("Tenant already linked to another domain")


async def get_installation(tenant_id: str) -> dict | None:
    db = get_db()
    return await db.google_chat_installations.find_one({"tenant_id": tenant_id})


async def set_scopes(tenant_id: str, space_names: list[str], allow_dm: bool) -> None:
    db = get_db()
    await db.google_chat_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"spaces": space_names, "allow_direct_messages": allow_dm}},
    )


async def list_spaces_cached(tenant_id: str) -> list[dict]:
    db = get_db()
    installation = await get_installation(tenant_id)
    if installation and installation.get("spaces_cache_at"):
        age = (_utcnow() - installation["spaces_cache_at"]).total_seconds()
        if installation.get("spaces_cache") and age < settings.google_chat_space_cache_seconds:
            return installation["spaces_cache"]

    token = await get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get("https://chat.googleapis.com/v1/spaces", headers=headers)
    response.raise_for_status()
    data = response.json()
    spaces = [{"name": s["name"], "display_name": s.get("displayName")} for s in data.get("spaces", [])]
    if installation:
        await db.google_chat_installations.update_one(
            {"tenant_id": tenant_id},
            {"$set": {"spaces_cache": spaces, "spaces_cache_at": _utcnow()}},
        )
    return spaces


async def license_allows_capture(tenant_id: str) -> bool:
    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        return False
    status = evaluate_license_status(license_doc)["status"]
    return status not in {"expired", "grace", "suspended"}


def is_decision_signal(text: str) -> bool:
    lowered = text.lower()
    keywords = ["decision:", "final decision", "resolved", "we decided", "decision made"]
    return any(keyword in lowered for keyword in keywords)


def is_thread_message(message: dict) -> bool:
    return bool(message.get("thread", {}).get("name"))


def build_decision_record(message: dict, tenant_id: str) -> dict:
    return {
        "tenant_id": tenant_id,
        "source": "google_chat",
        "space": message.get("space", {}).get("name"),
        "thread_id": message.get("thread", {}).get("name"),
        "message_id": message.get("name"),
        "text": message.get("text"),
        "created_at": _utcnow(),
    }


def parse_event_payload(raw_body: bytes) -> dict:
    return json.loads(raw_body.decode("utf-8"))


async def record_webhook_event(event_id: str) -> bool:
    db = get_db()
    existing = await db.google_chat_webhook_events.find_one({"event_id": event_id})
    if existing:
        return False
    await db.google_chat_webhook_events.insert_one({"event_id": event_id, "created_at": _utcnow()})
    return True


async def increment_thread_activity(tenant_id: str, thread_id: str, space: str | None) -> None:
    db = get_db()
    await db.google_chat_thread_activity.update_one(
        {"tenant_id": tenant_id, "thread_id": thread_id},
        {
            "$inc": {"message_count": 1},
            "$set": {"last_message_at": _utcnow(), "space": space},
        },
        upsert=True,
    )


def verify_webhook_signature(raw_body: bytes, signature: str, timestamp: str) -> bool:
    if not settings.google_chat_webhook_secret:
        return True
    if not signature or not timestamp:
        return False
    base = f"{timestamp}:{raw_body.decode('utf-8')}"
    expected = hmac.new(
        settings.google_chat_webhook_secret.encode("utf-8"),
        base.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
