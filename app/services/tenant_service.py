from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.db.mongo import get_db
from app.utils.serialize import serialize_doc


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


async def get_tenant(tenant_id: str) -> dict | None:
    db = get_db()
    doc = await db.tenants.find_one({"_id": _oid(tenant_id)})
    return serialize_doc(doc) if doc else None


async def list_tenants(limit: int = 100) -> list[dict]:
    db = get_db()
    cursor = db.tenants.find({}).sort("created_at", -1).limit(limit)
    return [serialize_doc(doc) async for doc in cursor]


async def create_tenant(name: str) -> dict:
    db = get_db()
    base_slug = _slugify(name)
    slug = base_slug
    suffix = 1
    while await db.tenants.find_one({"slug": slug}):
        suffix += 1
        slug = f"{base_slug}-{suffix}"
    doc = {"name": name, "slug": slug, "created_at": _utcnow()}
    result = await db.tenants.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize_doc(doc)


async def update_tenant(tenant_id: str, updates: dict) -> dict | None:
    db = get_db()
    update_fields = {k: v for k, v in updates.items() if v is not None}
    if not update_fields:
        return None
    doc = await db.tenants.find_one_and_update(
        {"_id": _oid(tenant_id)},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )
    return serialize_doc(doc) if doc else None


async def delete_tenant(tenant_id: str, deleted_by: str | None = None) -> dict | None:
    db = get_db()
    updates = {"deleted_at": _utcnow()}
    if deleted_by:
        updates["deleted_by"] = _oid(deleted_by)
    doc = await db.tenants.find_one_and_update(
        {"_id": _oid(tenant_id)},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    return serialize_doc(doc) if doc else None
