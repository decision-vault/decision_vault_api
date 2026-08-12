from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool
import logging

from app.core.config import settings
from app.middleware.auth import get_current_user
from app.middleware.guard import withGuard
from app.schemas.project_invite import (
    ProjectInviteAccepted,
    ProjectInviteCreate,
    ProjectInviteCreateResponse,
    ProjectInviteOut,
    ProjectInviteToken,
    ProjectMemberOut,
    ProjectMemberRoleUpdate,
)
from app.services.audit_service import log_event
from app.services.email_service import send_project_invite_email
from app.services.license_service import QuotaExceededError
from app.services.project_team_service import (
    accept_project_invite,
    create_project_invite,
    decline_project_invite,
    leave_project,
    list_project_invites,
    list_project_members,
    remove_project_member,
    revoke_project_invite,
    update_project_member_role,
)
from app.db.mongo import get_db

router = APIRouter(prefix="/api", tags=["project-team"])
logger = logging.getLogger("decisionvault.project_team")


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


def _normalize_member(doc: dict) -> dict:
    if not doc:
        return doc
    doc = _json_safe(doc)
    if "joined_at" in doc and doc.get("joined_at") is not None:
        doc["joined_at"] = doc["joined_at"].replace(tzinfo=None)
    return doc


def _actor_project_role(request: Request) -> str:
    return getattr(request.state, "project_role", None) or "project_admin"


@router.post("/projects/{project_id}/members/invites", response_model=ProjectInviteCreateResponse)
async def create_project_invite_route(
    project_id: str,
    payload: ProjectInviteCreate,
    request: Request,
    background: BackgroundTasks,
    user=Depends(withGuard(feature="project.invites.manage", projectRole="project_admin")),
):
    db = get_db()
    actor = await db.users.find_one({"_id": ObjectId(user.get("user_id"))}, {"email": 1, "role": 1})
    actor_email = (actor or {}).get("email") or ""

    try:
        invite_doc, raw_token = await create_project_invite(
            tenant_id=request.state.tenant_id,
            project_id=project_id,
            actor_user_id=user.get("user_id"),
            actor_email=actor_email,
            actor_project_role=_actor_project_role(request),
            email=str(payload.email),
            role=payload.role,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except QuotaExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    invite_link = f"{settings.frontend_base_url.rstrip('/')}{settings.project_invite_frontend_path}?token={raw_token}"

    async def _send_and_mark() -> None:
        try:
            await run_in_threadpool(
                send_project_invite_email,
                to_email=str(payload.email),
                invite_link=invite_link,
                inviter_email=actor_email or None,
                role=invite_doc.get("role"),
                project_name=invite_doc.get("project_name"),
            )
            await db.project_invites.update_one(
                {"_id": invite_doc["_id"]},
                {"$set": {"last_sent_at": datetime.now(timezone.utc)}},
            )
        except Exception:
            logger.exception(
                "project_invite_email_failed tenant_id=%s project_id=%s email=%s",
                request.state.tenant_id,
                project_id,
                str(payload.email),
            )

    background.add_task(_send_and_mark)

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.invite.created",
        entity_type="project_invite",
        entity_id=str(invite_doc.get("_id")),
        metadata={"project_id": project_id, "email": str(payload.email), "role": invite_doc.get("role")},
    )

    return {"invite": _normalize_invite(invite_doc), "invite_link": invite_link}


@router.get("/projects/{project_id}/members/invites", response_model=list[ProjectInviteOut])
async def list_project_invites_route(
    project_id: str,
    request: Request,
    include_expired: bool = False,
    _guard=Depends(withGuard(feature="project.invites.manage", projectRole="project_admin")),
):
    try:
        invites = await list_project_invites(
            tenant_id=request.state.tenant_id, project_id=project_id, include_expired=include_expired
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [_normalize_invite(doc) for doc in invites]


@router.post("/projects/{project_id}/members/invites/{invite_id}/revoke")
async def revoke_project_invite_route(
    project_id: str,
    invite_id: str,
    request: Request,
    user=Depends(withGuard(feature="project.invites.manage", projectRole="project_admin")),
):
    if not ObjectId.is_valid(invite_id):
        raise HTTPException(status_code=400, detail="Invalid invite id")
    try:
        revoked = await revoke_project_invite(
            tenant_id=request.state.tenant_id,
            project_id=project_id,
            invite_id=invite_id,
            actor_user_id=user.get("user_id"),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409 if "accepted" in str(exc) else 404, detail=str(exc))

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.invite.revoked",
        entity_type="project_invite",
        entity_id=invite_id,
        metadata={"project_id": project_id},
    )
    return {"status": "revoked"} if revoked else {"status": "already_revoked"}


@router.get("/projects/{project_id}/members", response_model=list[ProjectMemberOut])
async def list_project_members_route(
    project_id: str,
    request: Request,
    _guard=Depends(withGuard(feature="project.read", projectRole="viewer")),
):
    try:
        members = await list_project_members(
            tenant_id=request.state.tenant_id, project_id=project_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [_normalize_member(doc) for doc in members]


@router.post("/projects/{project_id}/members/me/leave")
async def leave_project_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="project.read", projectRole="viewer")),
):
    try:
        await leave_project(
            tenant_id=request.state.tenant_id,
            project_id=project_id,
            user_id=user.get("user_id"),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.member.left",
        entity_type="project",
        entity_id=project_id,
    )
    return {"status": "left"}


@router.patch("/projects/{project_id}/members/{user_id}", response_model=ProjectMemberOut)
async def update_project_member_role_route(
    project_id: str,
    user_id: str,
    payload: ProjectMemberRoleUpdate,
    request: Request,
    user=Depends(withGuard(feature="project.members.manage", projectRole="project_admin")),
):
    try:
        updated = await update_project_member_role(
            tenant_id=request.state.tenant_id,
            project_id=project_id,
            actor_user_id=user.get("user_id"),
            actor_project_role=_actor_project_role(request),
            target_user_id=user_id,
            role=payload.role,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.member.role_changed",
        entity_type="project",
        entity_id=project_id,
        metadata={"target_user_id": user_id, "role": payload.role},
    )
    return _normalize_member(updated)


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_project_member_route(
    project_id: str,
    user_id: str,
    request: Request,
    user=Depends(withGuard(feature="project.members.manage", projectRole="project_admin")),
):
    try:
        await remove_project_member(
            tenant_id=request.state.tenant_id,
            project_id=project_id,
            actor_user_id=user.get("user_id"),
            actor_project_role=_actor_project_role(request),
            target_user_id=user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.member.removed",
        entity_type="project",
        entity_id=project_id,
        metadata={"target_user_id": user_id},
    )
    return {"status": "removed"}


@router.post("/invites/project/accept", response_model=ProjectInviteAccepted)
async def accept_project_invite_route(payload: ProjectInviteToken, user=Depends(get_current_user)):
    try:
        result = await accept_project_invite(token=payload.token, user=user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await log_event(
        tenant_id=result["tenant_id"],
        actor_id=user.get("user_id"),
        action="project.invite.accepted",
        entity_type="project",
        entity_id=result["project_id"],
    )
    return result


@router.post("/invites/project/decline")
async def decline_project_invite_route(payload: ProjectInviteToken, user=Depends(get_current_user)):
    try:
        await decline_project_invite(token=payload.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "declined"}
