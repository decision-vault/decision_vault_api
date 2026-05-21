from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.core.rbac import ORG_ROLE_ORDER, is_super_admin
from app.db.mongo import get_db


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _active_user_match() -> dict:
    return {"$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}]}


def _is_active(user: dict) -> bool:
    return user.get("is_active", True) is True


def _role_value(role: str | None) -> int:
    return ORG_ROLE_ORDER.get((role or "").lower(), 0)


async def list_org_users(*, tenant_id: str) -> list[dict]:
    db = get_db()
    cursor = db.users.find(
        {"tenant_id": _oid(tenant_id), **_active_user_match()},
        {"email": 1, "role": 1, "provider": 1, "created_at": 1, "last_login_at": 1, "is_active": 1},
    ).sort("created_at", 1)
    return [doc async for doc in cursor]


async def set_org_user_active(
    *,
    tenant_id: str,
    actor_user_id: str,
    target_user_id: str,
    is_active: bool,
) -> dict:
    if actor_user_id == target_user_id:
        raise PermissionError("Cannot change your own status")

    db = get_db()
    actor = await db.users.find_one({"_id": _oid(actor_user_id)})
    if not actor:
        raise PermissionError("Actor not found")

    if str(actor.get("tenant_id")) != tenant_id:
        raise PermissionError("Tenant mismatch")

    if not _is_active(actor) or actor.get("deleted_at"):
        raise PermissionError("Actor inactive")

    actor_role = (actor.get("role") or "").lower()
    if not is_super_admin(actor_role) and _role_value(actor_role) < _role_value("admin"):
        raise PermissionError("Insufficient org role")

    target = await db.users.find_one({"_id": _oid(target_user_id), "tenant_id": _oid(tenant_id), **_active_user_match()})
    if not target:
        raise ValueError("User not found")

    target_role = (target.get("role") or "").lower()
    if not is_super_admin(actor_role) and _role_value(actor_role) <= _role_value(target_role):
        raise PermissionError("Cannot change status for this user")

    updated = await db.users.find_one_and_update(
        {"_id": target["_id"]},
        {"$set": {"is_active": bool(is_active)}},
        return_document=ReturnDocument.AFTER,
    )
    if is_active is False:
        await db.refresh_tokens.update_many({"user_id": target["_id"], "revoked": False}, {"$set": {"revoked": True}})
    return updated or target


async def delete_org_user(
    *,
    tenant_id: str,
    actor_user_id: str,
    target_user_id: str,
) -> bool:
    if actor_user_id == target_user_id:
        raise PermissionError("Cannot delete your own account")

    db = get_db()
    actor = await db.users.find_one({"_id": _oid(actor_user_id)})
    if not actor:
        raise PermissionError("Actor not found")

    if str(actor.get("tenant_id")) != tenant_id:
        raise PermissionError("Tenant mismatch")

    if not _is_active(actor) or actor.get("deleted_at"):
        raise PermissionError("Actor inactive")

    actor_role = (actor.get("role") or "").lower()
    if not is_super_admin(actor_role) and _role_value(actor_role) < _role_value("admin"):
        raise PermissionError("Insufficient org role")

    target = await db.users.find_one({"_id": _oid(target_user_id), "tenant_id": _oid(tenant_id), **_active_user_match()})
    if not target:
        raise ValueError("User not found")

    target_role = (target.get("role") or "").lower()
    if not is_super_admin(actor_role) and _role_value(actor_role) <= _role_value(target_role):
        raise PermissionError("Cannot delete this user")

    now = _utcnow()
    result = await db.users.update_one(
        {"_id": target["_id"], **_active_user_match()},
        {"$set": {"deleted_at": now, "deleted_by": _oid(actor_user_id), "is_active": False}},
    )
    if result.modified_count != 1:
        return False

    await db.refresh_tokens.update_many({"user_id": target["_id"], "revoked": False}, {"$set": {"revoked": True}})
    await db.project_members.update_many(
        {"tenant_id": _oid(tenant_id), "user_id": target["_id"], "deleted_at": None},
        {"$set": {"deleted_at": now, "deleted_by": _oid(actor_user_id)}},
    )
    return True
