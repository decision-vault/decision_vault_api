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


async def create_project(tenant_id: str, payload: dict) -> dict:
    db = get_db()
    base_slug = _slugify(payload["name"])
    slug = base_slug
    suffix = 1
    while await db.projects.find_one({"tenant_id": _oid(tenant_id), "slug": slug, "deleted_at": None}):
        suffix += 1
        slug = f"{base_slug}-{suffix}"
    doc = {
        "tenant_id": _oid(tenant_id),
        "name": payload["name"],
        "slug": slug,
        "description": payload.get("description"),
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "deleted_at": None,
    }
    result = await db.projects.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize_doc(doc)


async def list_projects(tenant_id: str, project_ids: list[str] | None = None) -> list[dict]:
    db = get_db()
    query: dict = {"tenant_id": _oid(tenant_id), "deleted_at": None}
    if project_ids:
        query["_id"] = {"$in": [_oid(pid) for pid in project_ids]}
    cursor = db.projects.find(query).sort("created_at", -1)
    return [serialize_doc(doc) async for doc in cursor]


async def get_project(tenant_id: str, project_id: str) -> dict | None:
    db = get_db()
    doc = await db.projects.find_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None}
    )
    return serialize_doc(doc) if doc else None


async def update_project(tenant_id: str, project_id: str, updates: dict) -> dict | None:
    db = get_db()
    update_fields = {k: v for k, v in updates.items() if v is not None}
    if not update_fields:
        return None
    update_fields["updated_at"] = _utcnow()
    doc = await db.projects.find_one_and_update(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )
    return serialize_doc(doc) if doc else None


async def delete_project(tenant_id: str, project_id: str, deleted_by: str) -> bool:
    db = get_db()
    result = await db.projects.update_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None},
        {"$set": {"deleted_at": _utcnow(), "deleted_by": _oid(deleted_by), "updated_at": _utcnow()}},
    )
    return result.modified_count == 1
