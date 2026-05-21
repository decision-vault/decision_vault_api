from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
import logging

from app.core.rbac import is_super_admin
from app.core.config import settings
from app.middleware.auth import get_current_user
from app.middleware.guard import withGuard
from app.schemas.tenant import TenantCreate, TenantOut, TenantUpdate
from app.schemas.org_invite import OrgInviteAccept, OrgInviteCreate, OrgInviteCreateResponse, OrgInviteOut
from app.schemas.org_user import OrgUserOut, OrgUserUpdate
from app.services.email_service import send_org_invite_email
from app.services.audit_service import log_event
from app.services.tenant_service import create_tenant, delete_tenant, get_tenant, list_tenants, update_tenant
from app.services.org_invite_service import accept_org_invite, create_org_invite, list_org_invites
from app.services.org_user_service import delete_org_user, list_org_users, set_org_user_active
from app.db.mongo import get_db


router = APIRouter(prefix="/api/orgs", tags=["orgs"])
logger = logging.getLogger("decisionvault.org_invites")


def _normalize(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    return doc


def _json_safe(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _normalize_invite(doc: dict) -> dict:
    if not doc:
        return doc
    doc = _json_safe(doc)
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    doc["status"] = doc.get("status") or "pending"
    return doc


def _normalize_user(doc: dict) -> dict:
    if not doc:
        return doc
    doc = _json_safe(doc)
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    doc["is_active"] = doc.get("is_active", True) is True
    return doc


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key="dv_refresh",
        value=refresh_token,
        httponly=True,
        secure=settings.secure_cookies,
        samesite=settings.cookie_samesite,
        max_age=settings.refresh_token_days * 24 * 60 * 60,
        domain=settings.cookie_domain,
        path="/api/auth",
    )


@router.get("/me", response_model=TenantOut)
async def get_org(
    request: Request,
    user=Depends(withGuard(feature="view_decision", orgRole="viewer")),
):
    tenant = await get_tenant(request.state.tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Organization not found")
    return _normalize(tenant)


@router.patch("/me", response_model=TenantOut)
async def update_org(
    payload: TenantUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    updated = await update_tenant(request.state.tenant_id, payload.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Organization not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.updated",
        entity_type="tenant",
        entity_id=request.state.tenant_id,
    )
    return _normalize(updated)


@router.delete("/me")
async def delete_org(
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    deleted = await delete_tenant(request.state.tenant_id, user.get("user_id"))
    if not deleted:
        raise HTTPException(status_code=404, detail="Organization not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.deleted",
        entity_type="tenant",
        entity_id=request.state.tenant_id,
    )
    return {"status": "deleted"}


@router.get("", response_model=list[TenantOut])
async def list_orgs(
    q: str | None = None,
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Forbidden")
    tenants = await list_tenants(search=q)
    return [_normalize(doc) for doc in tenants]


@router.post("", response_model=TenantOut)
async def create_org(payload: TenantCreate, user=Depends(get_current_user)):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Forbidden")
    tenant = await create_tenant(payload.name)
    return _normalize(tenant)


@router.get("/me/invites", response_model=list[OrgInviteOut])
async def list_org_invites_route(
    request: Request,
    include_expired: bool = False,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="admin")),
):
    invites = await list_org_invites(tenant_id=request.state.tenant_id, include_expired=include_expired)
    return [_normalize_invite(doc) for doc in invites]


@router.post("/me/invites", response_model=OrgInviteCreateResponse)
async def create_org_invite_route(
    payload: OrgInviteCreate,
    request: Request,
    background: BackgroundTasks,
    user=Depends(withGuard(feature="edit_decision", orgRole="admin")),
):
    db = get_db()
    inviter = await db.users.find_one({"_id": ObjectId(user.get("user_id"))}, {"email": 1, "role": 1})
    inviter_email = (inviter or {}).get("email")
    inviter_role = (inviter or {}).get("role") or user.get("role")

    project_access = None
    if payload.project_access:
        project_access = [item.model_dump() for item in payload.project_access]

    try:
        invite_doc, raw_token = await create_org_invite(
            tenant_id=request.state.tenant_id,
            inviter_user_id=user.get("user_id"),
            inviter_role=inviter_role,
            email=str(payload.email),
            role=payload.role,
            project_access=project_access,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    tenant = await get_tenant(request.state.tenant_id)
    org_name = (tenant or {}).get("name")
    invite_link = f"{settings.frontend_base_url.rstrip('/')}{settings.org_invite_frontend_path}?token={raw_token}"

    async def _send_and_mark() -> None:
        try:
            projects = []
            project_access = invite_doc.get("project_access") or []
            project_ids = []
            for entry in project_access:
                pid = entry.get("project_id")
                if pid:
                    project_ids.append(pid)
            if project_ids:
                cursor = db.projects.find(
                    {"tenant_id": ObjectId(request.state.tenant_id), "_id": {"$in": project_ids}},
                    {"name": 1},
                )
                async for p in cursor:
                    if p.get("name"):
                        projects.append(p["name"])
            await run_in_threadpool(
                send_org_invite_email,
                to_email=str(payload.email),
                invite_link=invite_link,
                inviter_email=inviter_email,
                role=invite_doc.get("role"),
                org_name=org_name,
                projects=projects or None,
            )
            await db.org_invites.update_one(
                {"_id": invite_doc["_id"]},
                {"$set": {"last_sent_at": datetime.now(timezone.utc)}},
            )
        except Exception:
            logger.exception("org_invite_email_failed tenant_id=%s email=%s", request.state.tenant_id, str(payload.email))
            # Email failures should not prevent returning the invite link.
            return

    background.add_task(_send_and_mark)

    invite_id = str(invite_doc.get("_id"))
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.invite.created",
        entity_type="org_invite",
        entity_id=invite_id,
        metadata={"email": str(payload.email), "role": invite_doc.get("role")},
    )

    return {"invite": _normalize_invite(invite_doc), "invite_link": invite_link}


@router.post("/me/invites/{invite_id}/reinvite", response_model=OrgInviteCreateResponse)
async def reinvite_org_invite_route(
    invite_id: str,
    request: Request,
    background: BackgroundTasks,
    user=Depends(withGuard(feature="edit_decision", orgRole="admin")),
):
    if not ObjectId.is_valid(invite_id):
        raise HTTPException(status_code=400, detail="Invalid invite id")

    db = get_db()
    inviter = await db.users.find_one({"_id": ObjectId(user.get("user_id"))}, {"email": 1, "role": 1})
    inviter_email = (inviter or {}).get("email")
    inviter_role = (inviter or {}).get("role") or user.get("role")

    existing = await db.org_invites.find_one({"_id": ObjectId(invite_id)})
    if not existing:
        raise HTTPException(status_code=404, detail="Invite not found")
    if str(existing.get("tenant_id")) != request.state.tenant_id:
        raise HTTPException(status_code=404, detail="Invite not found")

    project_access = existing.get("project_access") or None
    if project_access:
        project_access = [
            {
                "project_id": str(item.get("project_id") or ""),
                "project_role": str(item.get("project_role") or "contributor"),
            }
            for item in project_access
        ]

    try:
        invite_doc, raw_token = await create_org_invite(
            tenant_id=request.state.tenant_id,
            inviter_user_id=user.get("user_id"),
            inviter_role=inviter_role,
            email=str(existing.get("email") or ""),
            role=str(existing.get("role") or "member"),
            project_access=project_access,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    tenant = await get_tenant(request.state.tenant_id)
    org_name = (tenant or {}).get("name")
    invite_link = f"{settings.frontend_base_url.rstrip('/')}{settings.org_invite_frontend_path}?token={raw_token}"

    async def _send_and_mark() -> None:
        try:
            projects = []
            project_access = invite_doc.get("project_access") or []
            project_ids = []
            for entry in project_access:
                pid = entry.get("project_id")
                if pid:
                    project_ids.append(pid)
            if project_ids:
                cursor = db.projects.find(
                    {"tenant_id": ObjectId(request.state.tenant_id), "_id": {"$in": project_ids}},
                    {"name": 1},
                )
                async for p in cursor:
                    if p.get("name"):
                        projects.append(p["name"])
            await run_in_threadpool(
                send_org_invite_email,
                to_email=str(invite_doc.get("email") or ""),
                invite_link=invite_link,
                inviter_email=inviter_email,
                role=invite_doc.get("role"),
                org_name=org_name,
                projects=projects or None,
            )
            await db.org_invites.update_one(
                {"_id": invite_doc["_id"]},
                {"$set": {"last_sent_at": datetime.now(timezone.utc)}},
            )
        except Exception:
            logger.exception(
                "org_invite_reinvite_email_failed tenant_id=%s email=%s",
                request.state.tenant_id,
                str(invite_doc.get("email") or ""),
            )
            return

    background.add_task(_send_and_mark)

    new_invite_id = str(invite_doc.get("_id"))
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.invite.reinvited",
        entity_type="org_invite",
        entity_id=new_invite_id,
        metadata={
            "email": str(invite_doc.get("email") or ""),
            "role": invite_doc.get("role"),
            "source_invite_id": invite_id,
        },
    )

    return {"invite": _normalize_invite(invite_doc), "invite_link": invite_link}


@router.post("/invites/accept")
async def accept_org_invite_route(payload: OrgInviteAccept, response: Response):
    try:
        result = await accept_org_invite(token=payload.token, password=payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    user = result.get("user") or {}
    tenant_id = str(user.get("tenant_id") or "")
    user_id = str(user.get("_id") or "")
    if tenant_id and user_id and tenant_id != "None" and user_id != "None":
        await log_event(
            tenant_id=tenant_id,
            actor_id=user_id,
            action="org.invite.accepted",
            entity_type="user",
            entity_id=user_id,
        )

    _set_refresh_cookie(response, result["refresh_token"])
    safe_user = {
        "id": user_id,
        "tenant_id": tenant_id,
        "email": user.get("email", ""),
        "role": user.get("role", ""),
        "provider": user.get("provider", ""),
        "last_login_at": user.get("last_login_at"),
    }
    return {"access_token": result["access_token"], "expires_in": result["expires_in"], "user": safe_user}


@router.get("/me/users", response_model=list[OrgUserOut])
async def list_org_users_route(
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="admin")),
):
    users = await list_org_users(tenant_id=request.state.tenant_id)
    return [_normalize_user(doc) for doc in users]


@router.patch("/me/users/{user_id}", response_model=OrgUserOut)
async def update_org_user_route(
    user_id: str,
    payload: OrgUserUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="admin")),
):
    try:
        updated = await set_org_user_active(
            tenant_id=request.state.tenant_id,
            actor_user_id=user.get("user_id"),
            target_user_id=user_id,
            is_active=payload.is_active,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.user.activated" if payload.is_active else "org.user.deactivated",
        entity_type="user",
        entity_id=user_id,
        metadata={"is_active": bool(payload.is_active)},
    )

    return _normalize_user(updated)


@router.delete("/me/users/{user_id}")
async def delete_org_user_route(
    user_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="admin")),
):
    try:
        deleted = await delete_org_user(
            tenant_id=request.state.tenant_id,
            actor_user_id=user.get("user_id"),
            target_user_id=user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.user.deleted",
        entity_type="user",
        entity_id=user_id,
    )
    return {"status": "deleted"}
