from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import time

from slack_sdk.http_retry import RetryHandler
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

from app.core.config import settings
from app.db.mongo import get_db
from app.services.crypto_service import decrypt_token, encrypt_token
from app.services.license_service import evaluate_license_status, get_current_license


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
                "revoked_at": None,
                "revoked_reason": None,
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

    def can_retry(self, *, state, request, response, error=None, **kwargs):
        return response is not None and response.status_code >= 500

    def prepare_for_next_attempt(self, *, state, request, response, error=None, **kwargs):
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
        cache_at = _as_aware_utc(installation.get("channels_cache_at"))
        if cache_at:
            cache_age = (_utcnow() - cache_at).total_seconds()
    if installation.get("channels_cache") and cache_age is not None:
        if cache_age < settings.slack_channel_cache_seconds:
            return installation["channels_cache"]

    client = slack_client(installation_token(installation))
    try:
        response = client.conversations_list(types="public_channel,private_channel")
    except SlackApiError as exc:
        # If app is installed without private-channel scope, fallback to public channels.
        error_code = None
        needed_scope = None
        try:
            error_code = exc.response.get("error")
            needed_scope = exc.response.get("needed")
        except Exception:
            pass
        if error_code == "missing_scope" and needed_scope == "groups:read":
            response = client.conversations_list(types="public_channel")
        else:
            raise
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


async def list_channel_messages(tenant_id: str, channel_id: str, limit: int = 50) -> list[dict]:
    installation = await get_installation_by_tenant(tenant_id)
    if not installation:
        raise ValueError("Slack not installed")
    if installation.get("revoked_at"):
        raise ValueError("Slack install revoked")

    client = slack_client(installation_token(installation))
    try:
        response = client.conversations_history(channel=channel_id, limit=max(1, min(limit, 200)))
    except SlackApiError as exc:
        error_code = None
        try:
            error_code = exc.response.get("error")
        except Exception:
            pass
        if error_code == "not_in_channel":
            try:
                client.conversations_join(channel=channel_id)
                response = client.conversations_history(channel=channel_id, limit=max(1, min(limit, 200)))
            except SlackApiError as join_exc:
                join_error = None
                try:
                    join_error = join_exc.response.get("error")
                except Exception:
                    pass
                if join_error == "missing_scope":
                    raise ValueError(
                        "Slack token missing channels:join scope. Reconnect Slack integration and try again."
                    )
                if join_error in {"method_not_supported_for_channel_type", "not_in_channel"}:
                    raise ValueError(
                        "Bot is not in this channel. Invite the DecisionVault app to the channel and try again."
                    )
                raise ValueError(f"Slack history read failed: {join_error or 'unknown_error'}")
        else:
            raise ValueError(f"Slack history read failed: {error_code or 'unknown_error'}")
    messages = response.get("messages", [])
    result: list[dict] = []
    for msg in reversed(messages):
        result.append(
            {
                "id": msg.get("ts"),
                "text": msg.get("text") or "",
                "user": msg.get("user") or msg.get("bot_id") or "unknown",
                "created_at": msg.get("ts"),
            }
        )
    return result


async def post_channel_message(tenant_id: str, channel_id: str, text: str) -> dict:
    installation = await get_installation_by_tenant(tenant_id)
    if not installation:
        raise ValueError("Slack not installed")
    if installation.get("revoked_at"):
        raise ValueError("Slack install revoked")

    client = slack_client(installation_token(installation))
    try:
        response = client.chat_postMessage(channel=channel_id, text=text.strip())
    except SlackApiError as exc:
        error_code = None
        try:
            error_code = exc.response.get("error")
        except Exception:
            pass
        if error_code == "not_in_channel":
            # For public channels, join and retry once.
            try:
                client.conversations_join(channel=channel_id)
                response = client.chat_postMessage(channel=channel_id, text=text.strip())
            except SlackApiError as join_exc:
                join_error = None
                try:
                    join_error = join_exc.response.get("error")
                except Exception:
                    pass
                if join_error == "missing_scope":
                    raise ValueError(
                        "Slack token missing channels:join scope. Reconnect Slack integration and try again."
                    )
                if join_error in {"method_not_supported_for_channel_type", "not_in_channel"}:
                    raise ValueError(
                        "Bot is not in this channel. Invite the DecisionVault app to the channel and try again."
                    )
                raise ValueError(f"Slack message send failed: {join_error or 'unknown_error'}")
        else:
            raise ValueError(f"Slack message send failed: {error_code or 'unknown_error'}")
    return {
        "id": response.get("ts"),
        "text": text.strip(),
        "user": "you",
        "created_at": response.get("ts"),
    }
