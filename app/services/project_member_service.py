from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.db.mongo import get_db
from app.utils.serialize import serialize_doc


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def list_project_members(tenant_id: str, project_id: str) -> list[dict]:
    db = get_db()
    cursor = db.project_members.find(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "deleted_at": None,
        }
    )
    return [serialize_doc(doc) async for doc in cursor]


async def add_project_member(tenant_id: str, project_id: str, user_id: str, role: str) -> dict:
    db = get_db()
    doc = await db.project_members.find_one_and_update(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
        },
        {
            "$set": {"role": role, "deleted_at": None, "deleted_by": None},
            "$setOnInsert": {"created_at": _utcnow()},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return serialize_doc(doc)


async def update_project_member(
    tenant_id: str, project_id: str, user_id: str, role: str
) -> dict | None:
    db = get_db()
    doc = await db.project_members.find_one_and_update(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
            "deleted_at": None,
        },
        {"$set": {"role": role}},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        return None
    return serialize_doc(doc)


async def remove_project_member(
    tenant_id: str, project_id: str, user_id: str, deleted_by: str
) -> bool:
    db = get_db()
    result = await db.project_members.update_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
            "deleted_at": None,
        },
        {"$set": {"deleted_at": _utcnow(), "deleted_by": _oid(deleted_by)}},
    )
    return result.modified_count == 1


async def restore_project_member(
    tenant_id: str, project_id: str, user_id: str
) -> dict | None:
    db = get_db()
    doc = await db.project_members.find_one_and_update(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
        },
        {"$unset": {"deleted_at": "", "deleted_by": ""}},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        return None
    return serialize_doc(doc)
