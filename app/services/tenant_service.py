from datetime import datetime, timedelta, timezone

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


# Every collection that is scoped to a tenant and must be purged on hard delete.
TENANT_SCOPED_COLLECTIONS = [
    "activities",
    "audit_logs",
    "canvases",
    "comments",
    "documents",
    "licenses",
    "org_invites",
    "prd_generation_jobs",
    "project_invites",
    "project_members",
    "project_workflows",
    "projects",
    "refresh_tokens",
    "sprints",
    "tasks",
    "workflow_epics",
    "workflow_features",
    "workflow_phases",
    "workflow_sprints",
    "workflow_task_dependencies",
    "workflow_tasks",
    "workflows",
    "workspaces",
    "users",
]


async def get_tenant(tenant_id: str) -> dict | None:
    db = get_db()
    doc = await db.tenants.find_one({"_id": _oid(tenant_id)})
    return serialize_doc(doc) if doc else None


async def list_tenants(limit: int = 100, search: str | None = None) -> list[dict]:
    db = get_db()
    query: dict = {}
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    cursor = db.tenants.find(query).sort("created_at", -1).limit(limit)
    return [serialize_doc(doc) async for doc in cursor]


async def list_owned_tenants(user_id: str, limit: int = 100) -> list[dict]:
    db = get_db()
    cursor = (
        db.tenants.find({"owner_user_ids": _oid(user_id)})
        .sort("created_at", -1)
        .limit(limit)
    )
    return [serialize_doc(doc) async for doc in cursor]


async def user_owns_tenant(user_id: str, tenant_id: str) -> bool:
    db = get_db()
    doc = await db.tenants.find_one(
        {"_id": _oid(tenant_id), "owner_user_ids": _oid(user_id)},
        {"_id": 1},
    )
    return doc is not None


async def create_tenant(name: str, owner_user_id: str | None = None) -> dict:
    db = get_db()
    base_slug = _slugify(name)
    slug = base_slug
    suffix = 1
    while await db.tenants.find_one({"slug": slug}):
        suffix += 1
        slug = f"{base_slug}-{suffix}"
    doc = {
        "name": name,
        "slug": slug,
        "created_at": _utcnow(),
        "owner_user_ids": [_oid(owner_user_id)] if owner_user_id else [],
    }
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


async def restore_tenant(tenant_id: str) -> dict | None:
    db = get_db()
    doc = await db.tenants.find_one_and_update(
        {"_id": _oid(tenant_id)},
        {"$set": {"deleted_at": None, "deleted_by": None}},
        return_document=ReturnDocument.AFTER,
    )
    return serialize_doc(doc) if doc else None


async def list_deleted_tenants(limit: int = 200) -> list[dict]:
    db = get_db()
    cursor = db.tenants.find({"deleted_at": {"$ne": None}}).sort("deleted_at", -1).limit(limit)
    return [serialize_doc(doc) async for doc in cursor]


async def hard_delete_tenant(tenant_id: str) -> bool:
    db = get_db()
    oid = _oid(tenant_id)
    for collection_name in TENANT_SCOPED_COLLECTIONS:
        await db[collection_name].delete_many({"tenant_id": oid})
    result = await db.tenants.delete_one({"_id": oid})
    return result.deleted_count == 1


async def sweep_expired_deleted_tenants(grace_days: int) -> list[str]:
    db = get_db()
    cutoff = _utcnow() - timedelta(days=grace_days)
    removed: list[str] = []
    cursor = db.tenants.find({"deleted_at": {"$lte": cutoff}})
    async for doc in cursor:
        try:
            await hard_delete_tenant(str(doc["_id"]))
            removed.append(str(doc["_id"]))
        except Exception:
            continue
    return removed
