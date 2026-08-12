import json
import logging
from datetime import datetime

import redis.asyncio as redis
from bson import ObjectId

from app.core.config import settings
from app.db.mongo import get_db

logger = logging.getLogger("uvicorn.error")

NOTIFICATION_CHANNEL = "dv:notifications:{tenant_id}"

NOTIFICATION_TYPES = ("security", "performance", "messages", "system")

SEED_NOTIFICATIONS = [
    {
        "type": "messages",
        "title": "Welcome to DecisionVault",
        "message": "We're glad you're here. Start by creating your first project from the Projects page.",
        "severity": 3,
    },
    {
        "type": "performance",
        "title": "Usage tracking enabled",
        "message": "AI tokens and storage usage are now tracked live on the Usage page for your organization.",
        "severity": 2,
    },
    {
        "type": "security",
        "title": "Organization secured",
        "message": "Team permissions and org-level access controls are active for your workspace.",
        "severity": 1,
    },
]

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


def _oid(value) -> ObjectId:
    if isinstance(value, ObjectId):
        return value
    return ObjectId(value)


def serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = str(doc["_id"])
    doc["tenant_id"] = str(doc["tenant_id"])
    doc["user_id"] = str(doc["user_id"])
    if doc.get("project_id"):
        doc["project_id"] = str(doc["project_id"])
    doc.setdefault("is_read", False)
    doc.setdefault("severity", 2)
    if doc.get("created_at"):
        doc["created_at"] = doc["created_at"].isoformat()
    doc.pop("_id", None)
    return doc


async def _publish(tenant_id, event: dict) -> None:
    try:
        channel = NOTIFICATION_CHANNEL.format(tenant_id=str(tenant_id))
        await _get_redis().publish(channel, json.dumps(event, default=str))
    except Exception:
        logger.exception("notification_publish_failed")


async def _ensure_seeded(db, tenant_id: ObjectId, user_id: ObjectId) -> None:
    count = await db.notifications.count_documents(
        {"tenant_id": tenant_id, "user_id": user_id}
    )
    if count:
        return
    now = datetime.utcnow()
    docs = [
        {
            **item,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "project_id": None,
            "is_read": False,
            "created_at": now,
        }
        for item in SEED_NOTIFICATIONS
    ]
    if docs:
        await db.notifications.insert_many(docs)


async def create_notification(
    tenant_id: str,
    user_id: str,
    type_: str,
    title: str,
    message: str = "",
    severity: int = 2,
    project_id: str | None = None,
) -> dict:
    db = get_db()
    doc = {
        "tenant_id": _oid(tenant_id),
        "user_id": _oid(user_id),
        "type": type_,
        "title": title,
        "message": message,
        "severity": severity,
        "project_id": _oid(project_id) if project_id else None,
        "is_read": False,
        "created_at": datetime.utcnow(),
    }
    result = await db.notifications.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    serialized = serialize(doc)
    await _publish(tenant_id, serialized)
    return serialized


async def list_notifications(
    tenant_id: str,
    user_id: str,
    type_: str | None = None,
    status: str | None = None,
    severity: int | None = None,
    limit: int = 50,
) -> list[dict]:
    db = get_db()
    tenant_oid = _oid(tenant_id)
    user_oid = _oid(user_id)
    await _ensure_seeded(db, tenant_oid, user_oid)

    query = {"tenant_id": tenant_oid, "user_id": user_oid}
    if type_:
        query["type"] = type_
    if severity is not None:
        query["severity"] = severity
    if status == "read":
        query["is_read"] = True
    elif status == "unread":
        query["is_read"] = False

    cursor = db.notifications.find(query).sort("created_at", -1).limit(limit)
    return [serialize(doc) async for doc in cursor]


async def get_notification(tenant_id: str, user_id: str, notification_id: str) -> dict | None:
    if not ObjectId.is_valid(notification_id):
        return None
    db = get_db()
    return await db.notifications.find_one(
        {"_id": _oid(notification_id), "tenant_id": _oid(tenant_id), "user_id": _oid(user_id)}
    )


async def mark_read(tenant_id: str, user_id: str, notification_id: str) -> dict | None:
    db = get_db()
    updated = await db.notifications.find_one_and_update(
        {
            "_id": _oid(notification_id),
            "tenant_id": _oid(tenant_id),
            "user_id": _oid(user_id),
        },
        {"$set": {"is_read": True, "read_at": datetime.utcnow()}},
        return_document=True,
    )
    return serialize(updated) if updated else None


async def mark_all_read(tenant_id: str, user_id: str) -> int:
    db = get_db()
    result = await db.notifications.update_many(
        {"tenant_id": _oid(tenant_id), "user_id": _oid(user_id), "is_read": False},
        {"$set": {"is_read": True, "read_at": datetime.utcnow()}},
    )
    return result.modified_count


async def unread_count(tenant_id: str, user_id: str) -> int:
    db = get_db()
    return await db.notifications.count_documents(
        {"tenant_id": _oid(tenant_id), "user_id": _oid(user_id), "is_read": False}
    )
