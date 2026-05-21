from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from app.core.rbac import PROJECT_ROLE_ORDER, org_role_at_least
from app.db.mongo import get_db
from app.services.project_member_service import add_project_member


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _safe_id(value: ObjectId) -> str:
    return str(value)


def _active_user_match() -> dict:
    return {"$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}]}


def _is_active(user: dict) -> bool:
    return user.get("is_active", True) is True and user.get("deleted_at") is None


def _normalize_project_role(role: str) -> str:
    normalized = (role or "").strip().lower()
    return normalized if normalized in PROJECT_ROLE_ORDER else "contributor"


async def list_project_catalog(*, tenant_id: str) -> list[dict]:
    db = get_db()
    cursor = db.projects.find(
        {"tenant_id": _oid(tenant_id), "deleted_at": None},
        {"name": 1, "created_at": 1, "updated_at": 1, "last_used_at": 1, "deleted_at": 1},
    ).sort("created_at", -1).limit(200)
    projects: list[dict] = []
    async for doc in cursor:
        projects.append({"_id": doc["_id"], "name": doc.get("name") or "Project"})
    return projects


async def request_project_access(*, tenant_id: str, user_id: str, project_id: str) -> dict:
    db = get_db()

    project = await db.projects.find_one({"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None})
    if not project:
        raise ValueError("Project not found")

    user_doc = await db.users.find_one({"_id": _oid(user_id), "tenant_id": _oid(tenant_id), **_active_user_match()})
    if not user_doc or not _is_active(user_doc):
        raise PermissionError("User not active")

    membership = await db.project_members.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
            "deleted_at": None,
        }
    )
    if membership:
        raise ValueError("Already a project member")

    existing = await db.project_access_requests.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
            "status": "pending",
        }
    )
    if existing:
        return existing

    now = _utcnow()
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "project_name": project.get("name") or "Project",
        "user_id": _oid(user_id),
        "user_email": user_doc.get("email") or "",
        "status": "pending",
        "created_at": now,
        "decided_at": None,
        "decided_by_user_id": None,
    }
    result = await db.project_access_requests.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def list_access_requests(*, tenant_id: str, status: str = "pending") -> list[dict]:
    db = get_db()
    match = {"tenant_id": _oid(tenant_id)}
    if status:
        match["status"] = status
    cursor = db.project_access_requests.find(match).sort("created_at", -1).limit(200)
    return [doc async for doc in cursor]


async def list_my_access_requests(*, tenant_id: str, user_id: str, status: str = "pending") -> list[dict]:
    db = get_db()
    match = {"tenant_id": _oid(tenant_id), "user_id": _oid(user_id)}
    if status:
        match["status"] = status
    cursor = db.project_access_requests.find(match).sort("created_at", -1).limit(200)
    return [doc async for doc in cursor]


async def decide_access_request(
    *,
    tenant_id: str,
    actor_user_id: str,
    request_id: str,
    decision: str,
    default_role: str = "contributor",
) -> dict:
    if decision not in {"approved", "rejected"}:
        raise ValueError("Invalid decision")

    db = get_db()
    actor = await db.users.find_one({"_id": _oid(actor_user_id)})
    if not actor or str(actor.get("tenant_id")) != tenant_id:
        raise PermissionError("Forbidden")
    if not _is_active(actor):
        raise PermissionError("Forbidden")
    if not org_role_at_least((actor.get("role") or "").lower(), "admin"):
        raise PermissionError("Forbidden")

    now = _utcnow()
    req = await db.project_access_requests.find_one_and_update(
        {"_id": _oid(request_id), "tenant_id": _oid(tenant_id), "status": "pending"},
        {"$set": {"status": decision, "decided_at": now, "decided_by_user_id": _oid(actor_user_id)}},
        return_document=ReturnDocument.AFTER,
    )
    if not req:
        raise ValueError("Request not found")

    if decision == "approved":
        await add_project_member(
            tenant_id,
            _safe_id(req["project_id"]),
            _safe_id(req["user_id"]),
            _normalize_project_role(default_role),
        )
    return req


async def invite_user_to_project_by_email(
    *,
    tenant_id: str,
    actor_user_id: str,
    project_id: str,
    email: str,
    project_role: str,
) -> tuple[str | None, dict | None]:
    """
    Returns (user_id_added, org_invite_doc_created).
    - If user exists in tenant, adds project membership and returns user_id.
    - If user does not exist, creates no membership and returns None + invite payload info for caller.
    """
    db = get_db()
    actor = await db.users.find_one({"_id": _oid(actor_user_id)})
    if not actor or str(actor.get("tenant_id")) != tenant_id:
        raise PermissionError("Forbidden")
    if not _is_active(actor):
        raise PermissionError("Forbidden")
    if not org_role_at_least((actor.get("role") or "").lower(), "admin"):
        raise PermissionError("Forbidden")

    project = await db.projects.find_one({"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None})
    if not project:
        raise ValueError("Project not found")

    normalized_email = email.lower().strip()
    user_doc = await db.users.find_one({"tenant_id": _oid(tenant_id), "email": normalized_email, **_active_user_match()})
    if user_doc and not _is_active(user_doc):
        raise ValueError("User is deactivated or deleted")

    if user_doc and _is_active(user_doc):
        await add_project_member(tenant_id, project_id, _safe_id(user_doc["_id"]), _normalize_project_role(project_role))
        return _safe_id(user_doc["_id"]), None

    invite_payload = {
        "email": normalized_email,
        "role": "viewer",
        "project_access": [{"project_id": project_id, "project_role": _normalize_project_role(project_role)}],
    }
    return None, invite_payload
