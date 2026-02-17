from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from app.db.mongo import get_db
from app.services.crypto_service import encrypt_token_with_key, decrypt_token_with_key
from app.core.config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def store_oauth(tenant_id: str, provider: str, tokens: dict) -> None:
    db = get_db()
    await db.custom_oauth_tokens.update_one(
        {"tenant_id": tenant_id, "provider": provider},
        {
            "$set": {
                "tenant_id": tenant_id,
                "provider": provider,
                "access_token": encrypt_token_with_key(tokens["access_token"], settings.custom_connector_hmac_secret),
                "refresh_token": encrypt_token_with_key(tokens.get("refresh_token", ""), settings.custom_connector_hmac_secret)
                if tokens.get("refresh_token")
                else "",
                "expires_at": _utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
                "updated_at": _utcnow(),
            }
        },
        upsert=True,
    )


async def exchange_code(
    tenant_id: str,
    provider: str,
    token_url: str,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
    response.raise_for_status()
    tokens = response.json()
    await store_oauth(tenant_id, provider, tokens)
    return tokens


async def refresh_token(
    tenant_id: str, provider: str, token_url: str, client_id: str, client_secret: str
) -> str:
    db = get_db()
    doc = await db.custom_oauth_tokens.find_one({"tenant_id": tenant_id, "provider": provider})
    if not doc or not doc.get("refresh_token"):
        raise ValueError("Missing refresh token")
    refresh = decrypt_token_with_key(doc["refresh_token"], settings.custom_connector_hmac_secret)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
    response.raise_for_status()
    tokens = response.json()
    await store_oauth(tenant_id, provider, tokens)
    return tokens["access_token"]
