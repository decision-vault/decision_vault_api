from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import msal

from app.core.config import settings
from app.db.mongo import get_db
from app.services.crypto_service import decrypt_token_with_key, encrypt_token_with_key
from app.services.license_service import evaluate_license_status, get_current_license


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _msal_app():
    return msal.ConfidentialClientApplication(
        client_id=settings.teams_client_id,
        client_credential=settings.teams_client_secret,
        authority=f"https://login.microsoftonline.com/{settings.teams_tenant_id}",
    )


async def store_installation(tenant_id: str, aad_tenant_id: str, tokens: dict) -> None:
    db = get_db()
    await db.teams_installations.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "aad_tenant_id": aad_tenant_id,
                "access_token": encrypt_token_with_key(
                    tokens["access_token"], settings.teams_token_encryption_key
                ),
                "refresh_token": encrypt_token_with_key(
                    tokens["refresh_token"], settings.teams_token_encryption_key
                )
                if tokens.get("refresh_token")
                else "",
                "expires_at": _utcnow(),
                "installed_at": _utcnow(),
                "teams": [],
                "channels": [],
                "allow_private": False,
            }
        },
        upsert=True,
    )


async def get_installation(tenant_id: str) -> dict | None:
    db = get_db()
    return await db.teams_installations.find_one({"tenant_id": tenant_id})


async def get_installation_by_aad(aad_tenant_id: str) -> dict | None:
    db = get_db()
    return await db.teams_installations.find_one({"aad_tenant_id": aad_tenant_id})


async def ensure_tenant_org_binding(tenant_id: str, aad_tenant_id: str) -> None:
    db = get_db()
    existing_aad = await db.teams_installations.find_one({"aad_tenant_id": aad_tenant_id})
    if existing_aad and existing_aad.get("tenant_id") != tenant_id:
        raise ValueError("Azure AD tenant already linked to another DecisionVault tenant")

    existing_tenant = await db.teams_installations.find_one({"tenant_id": tenant_id})
    if existing_tenant and existing_tenant.get("aad_tenant_id") != aad_tenant_id:
        raise ValueError("DecisionVault tenant already linked to another Azure AD tenant")


async def mark_webhook_received(tenant_id: str, count: int) -> None:
    db = get_db()
    await db.teams_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"last_webhook_at": _utcnow(), "last_webhook_count": count}},
    )


async def mark_delta_sync(tenant_id: str, captured: int) -> None:
    db = get_db()
    await db.teams_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"last_delta_sync_at": _utcnow(), "last_delta_sync_captured": captured}},
    )


async def set_scopes(tenant_id: str, team_ids: list[str], channel_ids: list[str], allow_private: bool) -> None:
    db = get_db()
    await db.teams_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"teams": team_ids, "channels": channel_ids, "allow_private": allow_private}},
    )


async def license_allows_capture(tenant_id: str) -> bool:
    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        return False
    status = evaluate_license_status(license_doc)["status"]
    return status not in {"expired", "grace", "suspended"}


async def exchange_code_for_tokens(code: str) -> dict:
    app = _msal_app()
    result = app.acquire_token_by_authorization_code(
        code,
        scopes=["https://graph.microsoft.com/.default"],
        redirect_uri=settings.teams_redirect_uri,
    )
    if "access_token" not in result:
        raise ValueError("Token exchange failed")
    return {
        "access_token": result["access_token"],
        "refresh_token": result.get("refresh_token", ""),
    }


async def get_access_token(installation: dict) -> str:
    if installation.get("refresh_token"):
        app = _msal_app()
        result = app.acquire_token_by_refresh_token(
            decrypt_token_with_key(
                installation["refresh_token"], settings.teams_token_encryption_key
            ),
            scopes=["https://graph.microsoft.com/.default"],
        )
        if "access_token" in result:
            return result["access_token"]
    return decrypt_token_with_key(
        installation["access_token"], settings.teams_token_encryption_key
    )


async def graph_get(installation: dict, url: str) -> dict:
    token = await get_access_token(installation)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


async def get_message(
    installation: dict, team_id: str, channel_id: str, message_id: str
) -> dict:
    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages/{message_id}"
    return await graph_get(installation, url)


async def list_teams(installation: dict) -> list[dict]:
    data = await graph_get(installation, "https://graph.microsoft.com/v1.0/teams")
    return [{"id": t["id"], "name": t.get("displayName")} for t in data.get("value", [])]


async def list_channels(installation: dict, team_id: str) -> list[dict]:
    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels"
    data = await graph_get(installation, url)
    return [{"id": c["id"], "name": c.get("displayName")} for c in data.get("value", [])]


def is_decision_signal(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "decision:",
        "decision made",
        "we decided",
        "final decision",
        "resolved",
    ]
    return any(keyword in lowered for keyword in keywords)


def is_threaded_message(message: dict) -> bool:
    return bool(message.get("replyToId"))


def build_decision_record(message: dict, tenant_id: str) -> dict:
    return {
        "tenant_id": tenant_id,
        "source": "teams",
        "team_id": message.get("teamId"),
        "channel_id": message.get("channelIdentity", {}).get("channelId"),
        "message_id": message.get("id"),
        "thread_id": message.get("replyToId") or message.get("id"),
        "text": message.get("body", {}).get("content"),
        "created_at": _utcnow(),
    }


def parse_webhook_payload(raw_body: bytes) -> dict:
    return json.loads(raw_body.decode("utf-8"))
