from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from app.core.config import settings
from app.core.rbac import ORG_ROLE_ORDER, org_role_at_least
from app.db.mongo import get_db
from app.services.project_member_service import add_project_member
from app.utils.security import hash_password
from app.utils.token import create_access_token, create_refresh_token, hash_token


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime | None) -> datetime | None:
    if not value:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _safe_id(value: ObjectId) -> str:
    return str(value)


def _normalize_role(role: str) -> str:
    return (role or "").strip().lower()


def _invite_status(doc: dict) -> str:
    if doc.get("revoked_at"):
        return "revoked"
    if doc.get("accepted_at"):
        return "accepted"
    expires_at = _ensure_aware(doc.get("expires_at"))
    if expires_at and expires_at <= _utcnow():
        return "expired"
    return "pending"


def _can_grant_role(inviter_role: str | None, invited_role: str) -> bool:
    if not inviter_role:
        return False
    if invited_role not in ORG_ROLE_ORDER:
        return False
    return org_role_at_least(inviter_role, invited_role)


def _normalize_project_role(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"viewer", "contributor", "project_admin"}:
        return normalized
    return "contributor"


def _refresh_token_doc(
    *,
    user_id: ObjectId,
    tenant_id: ObjectId,
    jti: str,
    token_hash_value: str,
    expires_at: datetime,
) -> dict:
    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "jti": jti,
        "token_hash": token_hash_value,
        "created_at": _utcnow(),
        "expires_at": expires_at,
        "revoked": False,
        "replaced_by": None,
    }


async def create_org_invite(
    *,
    tenant_id: str,
    inviter_user_id: str,
    inviter_role: str | None,
    email: str,
    role: str,
    project_access: list[dict] | None = None,
) -> tuple[dict, str]:
    db = get_db()
    normalized_role = _normalize_role(role)
    if normalized_role not in ORG_ROLE_ORDER:
        raise ValueError("Invalid role")
    if not _can_grant_role(inviter_role, normalized_role):
        raise PermissionError("Cannot invite with requested role")

    tenant_oid = _oid(tenant_id)
    normalized_email = email.lower().strip()
    now = _utcnow()

    await db.org_invites.update_many(
        {
            "tenant_id": tenant_oid,
            "email": normalized_email,
            "accepted_at": None,
            "revoked_at": None,
            "expires_at": {"$gt": now},
        },
        {"$set": {"revoked_at": now}},
    )

    raw_token = secrets.token_urlsafe(32)
    token_hash_value = hash_token(raw_token)
    expires_at = now + timedelta(hours=settings.org_invite_expires_hours)
    normalized_project_access = []
    if project_access:
        for entry in project_access:
            project_id = str(entry.get("project_id") or "").strip()
            if ObjectId.is_valid(project_id):
                normalized_project_access.append(
                    {"project_id": _oid(project_id), "project_role": _normalize_project_role(entry.get("project_role"))}
                )
    doc = {
        "tenant_id": tenant_oid,
        "email": normalized_email,
        "role": normalized_role,
        "token_hash": token_hash_value,
        "created_by": _oid(inviter_user_id),
        "created_at": now,
        "expires_at": expires_at,
        "accepted_at": None,
        "accepted_by_user_id": None,
        "revoked_at": None,
        "last_sent_at": None,
        "project_access": normalized_project_access or None,
    }
    result = await db.org_invites.insert_one(doc)
    doc["_id"] = result.inserted_id
    doc["status"] = _invite_status(doc)
    return doc, raw_token


async def list_org_invites(*, tenant_id: str, include_expired: bool = False) -> list[dict]:
    db = get_db()
    match = {"tenant_id": _oid(tenant_id)}
    if not include_expired:
        match["revoked_at"] = None
        match["accepted_at"] = None
        match["expires_at"] = {"$gt": _utcnow()}

    cursor = db.org_invites.find(match).sort("created_at", -1).limit(200)
    invites: list[dict] = []
    async for doc in cursor:
        doc["status"] = _invite_status(doc)
        invites.append(doc)
    return invites


async def accept_org_invite(*, token: str, password: str | None) -> dict:
    db = get_db()
    now = _utcnow()
    token_hash_value = hash_token(token)

    invite = await db.org_invites.find_one({"token_hash": token_hash_value})
    if not invite:
        raise ValueError("Invalid invite token")
    if invite.get("revoked_at"):
        raise ValueError("Invite revoked")
    if invite.get("accepted_at"):
        raise ValueError("Invite already accepted")
    expires_at = _ensure_aware(invite.get("expires_at"))
    if expires_at and expires_at <= now:
        raise ValueError("Invite expired")

    tenant_oid: ObjectId = invite["tenant_id"]
    invited_role: str = invite["role"]
    invited_email: str = invite["email"]

    user = await db.users.find_one({"tenant_id": tenant_oid, "email": invited_email})
    if user and user.get("deleted_at") is not None:
        raise ValueError("Account deleted")
    if not user and not password:
        raise ValueError("Password required")

    if not user:
        user_doc = {
            "tenant_id": tenant_oid,
            "email": invited_email,
            "role": invited_role,
            "provider": "password",
            "password_hash": hash_password(password or ""),
            "is_active": True,
            "deleted_at": None,
            "deleted_by": None,
            "created_at": now,
            "last_login_at": None,
        }
        insert = await db.users.insert_one(user_doc)
        user_doc["_id"] = insert.inserted_id
        user = user_doc
    else:
        if user.get("is_active", True) is False:
            raise ValueError("Account deactivated")
        current_role = (user.get("role") or "").lower()
        if ORG_ROLE_ORDER.get(invited_role, 0) > ORG_ROLE_ORDER.get(current_role, 0):
            await db.users.update_one({"_id": user["_id"]}, {"$set": {"role": invited_role}})
            user["role"] = invited_role
        if password:
            await db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"password_hash": hash_password(password), "provider": "password"}},
            )

    await db.org_invites.update_one(
        {"_id": invite["_id"]},
        {"$set": {"accepted_at": now, "accepted_by_user_id": user["_id"]}},
    )

    project_access = invite.get("project_access") or []
    for entry in project_access:
        project_id = entry.get("project_id")
        if not project_id:
            continue
        try:
            project_oid = project_id if isinstance(project_id, ObjectId) else _oid(str(project_id))
        except Exception:
            continue
        project = await db.projects.find_one({"_id": project_oid, "tenant_id": tenant_oid, "deleted_at": None})
        if not project:
            continue
        await add_project_member(
            _safe_id(tenant_oid),
            _safe_id(project_oid),
            _safe_id(user["_id"]),
            _normalize_project_role(entry.get("project_role")),
        )

    access_token, expires_in = create_access_token(_safe_id(user["_id"]), _safe_id(tenant_oid), user["role"])
    refresh_token, jti, refresh_expires_at = create_refresh_token(
        _safe_id(user["_id"]), _safe_id(tenant_oid), user["role"]
    )
    await db.refresh_tokens.insert_one(
        _refresh_token_doc(
            user_id=user["_id"],
            tenant_id=tenant_oid,
            jti=jti,
            token_hash_value=hash_token(refresh_token),
            expires_at=refresh_expires_at,
        )
    )
    await db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login_at": now}})

    return {
        "user": user,
        "access_token": access_token,
        "expires_in": expires_in,
        "refresh_token": refresh_token,
        "refresh_jti": jti,
        "refresh_expires_at": refresh_expires_at,
    }
