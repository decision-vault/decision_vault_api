from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.db.mongo import get_db
from app.utils.serialize import serialize_doc

PAUSE_AFTER_DAYS = 30
RESTORE_DELAY_HOURS = 1


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: datetime | None) -> datetime:
    if not isinstance(value, datetime):
        return _utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _last_activity(doc: dict) -> datetime:
    candidate = doc.get("last_used_at") or doc.get("updated_at") or doc.get("created_at")
    return _as_aware_utc(candidate)


def _enrich_runtime_status(doc: dict) -> dict:
    now = _utcnow()
    last_activity = _last_activity(doc)
    idle_seconds = (now - last_activity).total_seconds()
    pause_after_seconds = PAUSE_AFTER_DAYS * 24 * 60 * 60
    if idle_seconds < pause_after_seconds:
        doc["status"] = "active"
        doc["status_message"] = "Project ready"
        doc["can_restore"] = False
        doc["restore_available_at"] = None
        return doc

    paused_at = last_activity.replace(microsecond=0) + timedelta(days=PAUSE_AFTER_DAYS)
    restore_available_at = paused_at + timedelta(hours=RESTORE_DELAY_HOURS)
    doc["status"] = "paused"
    doc["status_message"] = "Paused due to 30 days inactivity"
    doc["can_restore"] = now >= restore_available_at
    doc["restore_available_at"] = restore_available_at
    return doc


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
        "last_used_at": _utcnow(),
        "deleted_at": None,
    }
    result = await db.projects.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _enrich_runtime_status(serialize_doc(doc))


async def list_projects(
    tenant_id: str,
    project_ids: list[str] | None = None,
    search: str | None = None,
    status: str | None = None,
) -> list[dict]:
    db = get_db()
    query: dict = {"tenant_id": _oid(tenant_id), "deleted_at": None}
    if project_ids:
        query["_id"] = {"$in": [_oid(pid) for pid in project_ids]}
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    cursor = db.projects.find(query).sort("created_at", -1)
    projects = [_enrich_runtime_status(serialize_doc(doc)) async for doc in cursor]
    if status in {"active", "paused"}:
        return [doc for doc in projects if doc.get("status") == status]
    return projects


async def get_project(tenant_id: str, project_id: str) -> dict | None:
    db = get_db()
    doc = await db.projects.find_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None}
    )
    return _enrich_runtime_status(serialize_doc(doc)) if doc else None


async def update_project(tenant_id: str, project_id: str, updates: dict) -> dict | None:
    db = get_db()
    update_fields = {k: v for k, v in updates.items() if v is not None}
    if not update_fields:
        return None
    update_fields["updated_at"] = _utcnow()
    update_fields["last_used_at"] = _utcnow()
    doc = await db.projects.find_one_and_update(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )
    return _enrich_runtime_status(serialize_doc(doc)) if doc else None


async def delete_project(tenant_id: str, project_id: str, deleted_by: str) -> bool:
    db = get_db()
    result = await db.projects.update_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None},
        {"$set": {"deleted_at": _utcnow(), "deleted_by": _oid(deleted_by), "updated_at": _utcnow()}},
    )
    return result.modified_count == 1


async def restore_project(tenant_id: str, project_id: str) -> tuple[bool, str]:
    db = get_db()
    doc = await db.projects.find_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None}
    )
    if not doc:
        return False, "Project not found"

    runtime = _enrich_runtime_status(serialize_doc(doc))
    if runtime.get("status") != "paused":
        return False, "Project is not paused"
    if not runtime.get("can_restore"):
        return False, "Restore available after 1 hour from pause"

    result = await db.projects.update_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None},
        {"$set": {"last_used_at": _utcnow(), "updated_at": _utcnow()}},
    )
    if result.modified_count != 1:
        return False, "Failed to restore project"
    return True, "restored"
