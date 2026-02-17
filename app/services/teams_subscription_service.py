from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from app.core.config import settings
from app.db.mongo import get_db
from app.services.teams_service import get_access_token


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def create_subscription(installation: dict, notification_url: str) -> dict:
    token = await get_access_token(installation)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "changeType": "created",
        "notificationUrl": notification_url,
        "resource": "/teams/getAllMessages",
        "expirationDateTime": (_utcnow() + timedelta(days=1)).isoformat(),
        "clientState": "decisionvault",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://graph.microsoft.com/v1.0/subscriptions",
            headers=headers,
            json=payload,
        )
    response.raise_for_status()
    return response.json()


async def store_subscription(tenant_id: str, subscription: dict) -> None:
    db = get_db()
    await db.teams_subscriptions.update_one(
        {"tenant_id": tenant_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "subscription_id": subscription.get("id"),
                "resource": subscription.get("resource"),
                "expiration": subscription.get("expirationDateTime"),
                "created_at": _utcnow(),
            }
        },
        upsert=True,
    )


async def renew_subscription(installation: dict, subscription_id: str) -> dict:
    token = await get_access_token(installation)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "expirationDateTime": (_utcnow() + timedelta(days=1)).isoformat(),
    }
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"https://graph.microsoft.com/v1.0/subscriptions/{subscription_id}",
            headers=headers,
            json=payload,
        )
    response.raise_for_status()
    return response.json()


async def renew_due_subscriptions() -> int:
    db = get_db()
    now = _utcnow()
    renewed = 0
    cursor = db.teams_subscriptions.find({})
    async for sub in cursor:
        expiration = sub.get("expiration")
        if not expiration:
            continue
        if isinstance(expiration, str):
            try:
                expiration = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
            except Exception:
                continue
        if expiration - now > timedelta(hours=6):
            continue
        installation = await db.teams_installations.find_one({"tenant_id": sub["tenant_id"]})
        if not installation:
            continue
        updated = await renew_subscription(installation, sub["subscription_id"])
        await db.teams_subscriptions.update_one(
            {"tenant_id": sub["tenant_id"]},
            {"$set": {"expiration": updated.get("expirationDateTime")}},
        )
        renewed += 1
    return renewed
