from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from app.core.config import settings
from app.core.rbac import PROJECT_ROLE_ORDER, project_role_at_least
from app.db.mongo import get_db
from app.utils.token import hash_token

VALID_PROJECT_ROLES = {"viewer", "contributor", "project_admin"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime | None) -> datetime | None:
    if not value:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _oid(value: str | ObjectId) -> ObjectId:
    return ObjectId(value)


def _normalize_role(role: str | None) -> str:
    return (role or "").strip().lower()


def _invite_status(doc: dict) -> str:
    if doc.get("revoked_at"):
        return "revoked"
    if doc.get("declined_at"):
        return "declined"
    if doc.get("accepted_at"):
        return "accepted"
    expires_at = _ensure_aware(doc.get("expires_at"))
    if expires_at and expires_at <= _utcnow():
        return "expired"
    return "pending"


async def _require_project(db, tenant_id: str, project_id: str) -> dict:
    project = await db.projects.find_one(
        {"_id": _oid(project_id), "tenant_id": _oid(tenant_id), "deleted_at": None}
    )
    if not project:
        raise ValueError("Project not found")
    return project


def _active_user_match() -> dict:
    return {"$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}]}


async def _resolve_org_owner(db, tenant_id: str) -> ObjectId | None:
    for role in ("owner", "admin"):
        user = await db.users.find_one(
            {"tenant_id": _oid(tenant_id), "role": role, **_active_user_match()},
            sort=[("created_at", 1)],
        )
        if user:
            return user["_id"]
    user = await db.users.find_one(
        {"tenant_id": _oid(tenant_id), **_active_user_match()},
        sort=[("created_at", 1)],
    )
    return user["_id"] if user else None


async def _effective_owner_user_id(db, tenant_id: str, project: dict) -> str | None:
    if project.get("owner_id"):
        return str(project["owner_id"])
    owner = await _resolve_org_owner(db, tenant_id)
    return str(owner) if owner else None


async def create_project_invite(
    *,
    tenant_id: str,
    project_id: str,
    actor_user_id: str,
    actor_email: str,
    actor_project_role: str,
    email: str,
    role: str,
) -> tuple[dict, str]:
    db = get_db()
    normalized_role = _normalize_role(role)
    if normalized_role not in VALID_PROJECT_ROLES:
        raise ValueError("Invalid project role")

    normalized_email = email.lower().strip()
    if normalized_email == (actor_email or "").lower().strip():
        raise ValueError("You are already a member of this project")

    from app.services.license_service import QuotaExceededError, enforce_team_member_quota

    await enforce_team_member_quota(tenant_id, pending_email=normalized_email)

    if not project_role_at_least(actor_project_role, normalized_role):
        raise PermissionError("Cannot invite with a role higher than your own")

    project = await _require_project(db, tenant_id, project_id)

    existing_invitee = await db.users.find_one(
        {"tenant_id": _oid(tenant_id), "email": normalized_email, **_active_user_match()}
    )
    if existing_invitee:
        existing_membership = await db.project_members.find_one(
            {
                "project_id": _oid(project_id),
                "user_id": existing_invitee["_id"],
                "removed_at": None,
            }
        )
        if existing_membership:
            raise ValueError("User is already a member of this project")

    now = _utcnow()
    await db.project_invites.update_many(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "email": normalized_email,
            "revoked_at": None,
            "declined_at": None,
            "accepted_at": None,
        },
        {"$set": {"revoked_at": now, "status": "revoked"}},
    )

    raw_token = secrets.token_urlsafe(32)
    expires_at = now + timedelta(hours=settings.project_invite_expires_hours)
    doc = {
        "tenant_id": _oid(tenant_id),
        "project_id": _oid(project_id),
        "project_name": project.get("name"),
        "email": normalized_email,
        "invited_by": _oid(actor_user_id),
        "invited_by_email": actor_email,
        "role": normalized_role,
        "token_hash": hash_token(raw_token),
        "status": "pending",
        "created_at": now,
        "expires_at": expires_at,
        "accepted_at": None,
        "accepted_by_user_id": None,
        "declined_at": None,
        "revoked_at": None,
        "revoked_by": None,
    }
    result = await db.project_invites.insert_one(doc)
    doc["_id"] = result.inserted_id
    doc["status"] = _invite_status(doc)
    return doc, raw_token


async def list_project_invites(
    *, tenant_id: str, project_id: str, include_expired: bool = False
) -> list[dict]:
    db = get_db()
    await _require_project(db, tenant_id, project_id)
    match: dict = {"tenant_id": _oid(tenant_id), "project_id": _oid(project_id)}
    if not include_expired:
        match["revoked_at"] = None
        match["declined_at"] = None
        match["accepted_at"] = None
        match["expires_at"] = {"$gt": _utcnow()}

    cursor = db.project_invites.find(match).sort("created_at", -1).limit(200)
    invites: list[dict] = []
    async for doc in cursor:
        doc["status"] = _invite_status(doc)
        invites.append(doc)
    return invites


async def revoke_project_invite(
    *, tenant_id: str, project_id: str, invite_id: str, actor_user_id: str
) -> bool:
    db = get_db()
    await _require_project(db, tenant_id, project_id)
    invite = await db.project_invites.find_one({"_id": _oid(invite_id)})
    if not invite or str(invite.get("tenant_id")) != tenant_id:
        raise ValueError("Invite not found")

    status = _invite_status(invite)
    if status == "accepted":
        raise ValueError("Invite already accepted")
    if status == "revoked":
        return False

    now = _utcnow()
    await db.project_invites.update_one(
        {"_id": invite["_id"]},
        {"$set": {"revoked_at": now, "revoked_by": _oid(actor_user_id), "status": "revoked"}},
    )
    return True


async def _load_invite_for_token(db, raw_token: str) -> dict:
    invite = await db.project_invites.find_one({"token_hash": hash_token(raw_token)})
    if not invite:
        raise ValueError("Invalid invite token")
    status = _invite_status(invite)
    if status == "revoked":
        raise ValueError("Invite revoked")
    if status == "declined":
        raise ValueError("Invite declined")
    if status == "accepted":
        raise ValueError("Invite already accepted")
    if status == "expired":
        raise ValueError("Invite expired")
    return invite


async def _upsert_membership(db, *, tenant_id: str, project_id: str, user_id: str, role: str, actor_user_id: str) -> None:
    now = _utcnow()
    existing = await db.project_members.find_one(
        {"project_id": _oid(project_id), "user_id": _oid(user_id)}
    )
    if existing:
        await db.project_members.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "role": role,
                    "created_by": _oid(actor_user_id),
                    "updated_at": now,
                    "removed_at": None,
                }
            },
        )
        return
    await db.project_members.insert_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
            "role": role,
            "created_by": _oid(actor_user_id),
            "created_at": now,
            "updated_at": now,
            "removed_at": None,
        }
    )


async def accept_project_invite(*, token: str, user: dict) -> dict:
    db = get_db()
    invite = await _load_invite_for_token(db, token)
    tenant_id = str(invite["tenant_id"])
    project_id = str(invite["project_id"])
    invited_email = invite["email"]
    invited_role = invite["role"]

    now = _utcnow()
    user_id = user.get("user_id")
    if not user_id:
        raise ValueError("Please sign in to accept this invitation")

    actor = await db.users.find_one({"_id": _oid(user_id)})
    if not actor:
        raise ValueError("Account not found")

    actor_email = (actor.get("email") or "").lower().strip()
    if actor_email != invited_email.lower().strip():
        raise ValueError("This invite is for a different account")

    invitee = await db.users.find_one(
        {
            "tenant_id": invite["tenant_id"],
            "email": invited_email,
            **_active_user_match(),
        }
    )
    if invitee:
        if invitee["_id"] != _oid(user_id):
            raise ValueError("This invite is for a different account")
    else:
        invitee = {
            "tenant_id": invite["tenant_id"],
            "email": invited_email,
            "name": actor.get("name"),
            "role": "member",
            "provider": actor.get("provider") or "password",
            "is_active": True,
            "deleted_at": None,
            "deleted_by": None,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }
        insert = await db.users.insert_one(invitee)
        invitee["_id"] = insert.inserted_id

    await _upsert_membership(
        db,
        tenant_id=tenant_id,
        project_id=project_id,
        user_id=str(invitee["_id"]),
        role=invited_role,
        actor_user_id=str(invitee["_id"]),
    )

    await db.project_invites.update_one(
        {"_id": invite["_id"]},
        {
            "$set": {
                "accepted_at": now,
                "accepted_by_user_id": invitee["_id"],
                "status": "accepted",
            }
        },
    )

    project = await db.projects.find_one({"_id": invite["project_id"]}, {"name": 1})
    return {
        "tenant_id": tenant_id,
        "project_id": project_id,
        "project_name": (project or {}).get("name") or invite.get("project_name") or "Project",
        "role": invited_role,
    }


async def decline_project_invite(*, token: str) -> bool:
    db = get_db()
    invite = await _load_invite_for_token(db, token)
    await db.project_invites.update_one(
        {"_id": invite["_id"]},
        {"$set": {"declined_at": _utcnow(), "status": "declined"}},
    )
    return True


async def list_project_members(*, tenant_id: str, project_id: str) -> list[dict]:
    db = get_db()
    project = await _require_project(db, tenant_id, project_id)
    owner_id = project.get("owner_id")
    if not owner_id:
        owner_id = await _resolve_org_owner(db, tenant_id)

    cursor = db.project_members.find(
        {"tenant_id": _oid(tenant_id), "project_id": _oid(project_id), "removed_at": None}
    ).sort("created_at", 1)
    members: list[dict] = []
    user_ids = set()
    if owner_id:
        user_ids.add(str(owner_id))
    async for doc in cursor:
        user_ids.add(str(doc["user_id"]))
        members.append(doc)

    users: dict[str, dict] = {}
    if user_ids:
        cursor = db.users.find({"_id": {"$in": [_oid(uid) for uid in user_ids]}}, {"email": 1, "name": 1})
        async for u in cursor:
            users[str(u["_id"])] = u

    seen: set[str] = set()
    rows: list[dict] = []

    if owner_id:
        owner_str = str(owner_id)
        seen.add(owner_str)
        rows.append(
            {
                "user_id": owner_str,
                "email": (users.get(owner_str) or {}).get("email", ""),
                "name": (users.get(owner_str) or {}).get("name"),
                "role": "project_admin",
                "is_owner": True,
                "joined_at": None,
            }
        )

    for doc in members:
        uid = str(doc["user_id"])
        if uid in seen:
            continue
        seen.add(uid)
        rows.append(
            {
                "user_id": uid,
                "email": (users.get(uid) or {}).get("email", ""),
                "name": (users.get(uid) or {}).get("name"),
                "role": doc.get("role", "contributor"),
                "is_owner": False,
                "joined_at": _ensure_aware(doc.get("created_at")),
            }
        )
    return rows


async def _load_membership(db, *, tenant_id: str, project_id: str, user_id: str) -> dict | None:
    return await db.project_members.find_one(
        {
            "tenant_id": _oid(tenant_id),
            "project_id": _oid(project_id),
            "user_id": _oid(user_id),
            "removed_at": None,
        }
    )


async def update_project_member_role(
    *,
    tenant_id: str,
    project_id: str,
    actor_user_id: str,
    actor_project_role: str,
    target_user_id: str,
    role: str,
) -> dict:
    db = get_db()
    normalized_role = _normalize_role(role)
    if normalized_role not in VALID_PROJECT_ROLES:
        raise ValueError("Invalid project role")

    project = await _require_project(db, tenant_id, project_id)
    owner_user_id = await _effective_owner_user_id(db, tenant_id, project)
    if owner_user_id == target_user_id:
        raise PermissionError("Cannot change the project owner's role")

    membership = await _load_membership(db, tenant_id=tenant_id, project_id=project_id, user_id=target_user_id)
    if not membership:
        raise ValueError("Member not found")

    if not project_role_at_least(actor_project_role, normalized_role):
        raise PermissionError("Cannot grant a role higher than your own")

    current_role = membership.get("role", "contributor")
    if PROJECT_ROLE_ORDER.get(current_role, 0) >= PROJECT_ROLE_ORDER.get(actor_project_role, 0):
        raise PermissionError("Cannot change the role of a user with an equal or higher role")

    now = _utcnow()
    await db.project_members.update_one(
        {"_id": membership["_id"]},
        {"$set": {"role": normalized_role, "updated_at": now, "actor": _oid(actor_user_id)}},
    )
    return {"user_id": target_user_id, "role": normalized_role}


async def remove_project_member(
    *,
    tenant_id: str,
    project_id: str,
    actor_user_id: str,
    actor_project_role: str,
    target_user_id: str,
) -> bool:
    db = get_db()
    project = await _require_project(db, tenant_id, project_id)
    owner_user_id = await _effective_owner_user_id(db, tenant_id, project)
    if owner_user_id == target_user_id:
        raise PermissionError("Cannot remove the project owner")

    membership = await _load_membership(db, tenant_id=tenant_id, project_id=project_id, user_id=target_user_id)
    if not membership:
        raise ValueError("Member not found")

    current_role = membership.get("role", "contributor")
    if PROJECT_ROLE_ORDER.get(current_role, 0) >= PROJECT_ROLE_ORDER.get(actor_project_role, 0):
        raise PermissionError("Cannot remove a user with an equal or higher role")

    await db.project_members.update_one(
        {"_id": membership["_id"]},
        {"$set": {"removed_at": _utcnow(), "updated_at": _utcnow(), "removed_by": _oid(actor_user_id)}},
    )
    return True


async def leave_project(*, tenant_id: str, project_id: str, user_id: str) -> bool:
    db = get_db()
    project = await _require_project(db, tenant_id, project_id)
    owner_user_id = await _effective_owner_user_id(db, tenant_id, project)
    if owner_user_id == user_id:
        raise PermissionError("Transfer ownership before leaving")

    membership = await _load_membership(db, tenant_id=tenant_id, project_id=project_id, user_id=user_id)
    if not membership:
        raise ValueError("You are not a member of this project")

    await db.project_members.update_one(
        {"_id": membership["_id"]},
        {"$set": {"removed_at": _utcnow(), "updated_at": _utcnow(), "left_by": _oid(user_id)}},
    )
    return True
