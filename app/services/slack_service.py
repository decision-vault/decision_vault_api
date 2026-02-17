from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

from slack_sdk import WebClient
import time

from slack_sdk.http_retry import RetryHandler
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

from app.core.config import settings
from app.db.mongo import get_db
from app.services.crypto_service import decrypt_token, encrypt_token
from app.services.license_service import evaluate_license_status, get_current_license


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _verify_slack_signature(raw_body: bytes, timestamp: str, signature: str) -> bool:
    if not timestamp.isdigit():
        return False
    # Reject requests older than 5 minutes to prevent replay.
    now = int(_utcnow().timestamp())
    if abs(now - int(timestamp)) > 60 * 5:
        return False
    basestring = f"v0:{timestamp}:{raw_body.decode('utf-8')}"
    computed = (
        "v0="
        + hmac.new(
            settings.slack_signing_secret.encode("utf-8"),
            basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )
    return hmac.compare_digest(computed, signature)


async def store_installation(
    *, tenant_id: str, team_id: str, team_name: str | None, bot_token: str
) -> None:
    db = get_db()
    await db.slack_installations.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "team_id": team_id,
                "team_name": team_name,
                "bot_token": encrypt_token(bot_token),
                "installed_at": _utcnow(),
                "channels": [],
            }
        },
        upsert=True,
    )


async def set_channels(tenant_id: str, channels: list[str]) -> None:
    db = get_db()
    await db.slack_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"channels": channels}},
    )


async def get_installation_by_team(team_id: str) -> dict | None:
    db = get_db()
    return await db.slack_installations.find_one({"team_id": team_id})


def installation_token(installation: dict) -> str:
    return decrypt_token(installation["bot_token"])


async def ensure_tenant_workspace_binding(tenant_id: str, team_id: str) -> None:
    db = get_db()
    existing_team = await db.slack_installations.find_one({"team_id": team_id})
    if existing_team and existing_team.get("tenant_id") != tenant_id:
        raise ValueError("Slack workspace already linked to another tenant")

    existing_tenant = await db.slack_installations.find_one({"tenant_id": tenant_id})
    if existing_tenant and existing_tenant.get("team_id") != team_id:
        raise ValueError("Tenant already linked to another Slack workspace")


async def get_installation_by_tenant(tenant_id: str) -> dict | None:
    db = get_db()
    return await db.slack_installations.find_one({"tenant_id": tenant_id})


async def mark_token_revoked(team_id: str, reason: str) -> None:
    db = get_db()
    await db.slack_installations.update_one(
        {"team_id": team_id},
        {"$set": {"revoked_at": _utcnow(), "revoked_reason": reason}},
    )


async def revoke_installation(tenant_id: str, reason: str) -> None:
    db = get_db()
    await db.slack_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"revoked_at": _utcnow(), "revoked_reason": reason}},
    )


class SlackRetryHandler(RetryHandler):
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 4.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def can_retry(self, *, state, request, response, error):
        return response is not None and response.status_code >= 500

    def prepare_for_next_attempt(self, *, state, request, response, error):
        attempt = state.current_attempt
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        time.sleep(delay)
        state.next_attempt_requested = True


def slack_client(bot_token: str) -> WebClient:
    return WebClient(
        token=bot_token,
        retry_handlers=[RateLimitErrorRetryHandler(), SlackRetryHandler()],
    )


async def list_channels(installation: dict) -> list[dict]:
    cache_age = None
    if installation.get("channels_cache_at"):
        cache_age = (_utcnow() - installation["channels_cache_at"]).total_seconds()
    if installation.get("channels_cache") and cache_age is not None:
        if cache_age < settings.slack_channel_cache_seconds:
            return installation["channels_cache"]

    client = slack_client(installation_token(installation))
    response = client.conversations_list(types="public_channel,private_channel")
    channels = response.get("channels", [])
    result = [{"id": c["id"], "name": c.get("name")} for c in channels]
    db = get_db()
    await db.slack_installations.update_one(
        {"_id": installation["_id"]},
        {"$set": {"channels_cache": result, "channels_cache_at": _utcnow()}},
    )
    return result


async def is_channel_allowed(installation: dict, channel_id: str | None) -> bool:
    channels = installation.get("channels", [])
    if not channels:
        return False
    return channel_id in channels


async def license_allows_capture(tenant_id: str) -> bool:
    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        return False
    status = evaluate_license_status(license_doc)["status"]
    return status not in {"expired", "grace", "suspended"}


def build_decision_record(event: dict, tenant_id: str) -> dict:
    return {
        "tenant_id": tenant_id,
        "source": "slack",
        "channel_id": event.get("channel"),
        "thread_ts": event.get("thread_ts") or event.get("ts"),
        "message_ts": event.get("ts"),
        "user_id": event.get("user"),
        "text": event.get("text"),
        "created_at": _utcnow(),
    }


def is_decision_signal(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "decision:",
        "decided",
        "we decided",
        "final decision",
        "decision made",
    ]
    return any(keyword in lowered for keyword in keywords)


def is_thread_event(event: dict) -> bool:
    return bool(event.get("thread_ts"))


def parse_slash_payload(body: str) -> dict:
    from urllib.parse import parse_qs

    parsed = parse_qs(body)
    return {k: v[0] for k, v in parsed.items()}


def verify_request(raw_body: bytes, timestamp: str, signature: str) -> bool:
    if not timestamp or not signature:
        return False
    return _verify_slack_signature(raw_body, timestamp, signature)


def parse_event_payload(raw_body: bytes) -> dict:
    return json.loads(raw_body.decode("utf-8"))
