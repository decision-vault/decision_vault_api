from datetime import datetime, timezone

from bson import ObjectId

from app.db.mongo import get_db


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str | None) -> ObjectId | None:
    if not value:
        return None
    if not ObjectId.is_valid(value):
        return None
    return ObjectId(value)


async def log_event(
    *,
    tenant_id: str,
    actor_id: str,
    action: str,
    entity_type: str,
    entity_id: str,
    metadata: dict | None = None,
) -> None:
    db = get_db()
    doc = {
        "tenant_id": _oid(tenant_id),
        "actor_id": _oid(actor_id),
        "action": action,
        "entity_type": entity_type,
        "entity_id": _oid(entity_id),
        "metadata": metadata or {},
        "created_at": _utcnow(),
    }
    await db.audit_logs.insert_one(doc)
