from datetime import datetime, timedelta, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request

from app.middleware.guard import withGuard
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate
from app.services.audit_service import log_event
from app.services.project_service import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    restore_project,
    update_project,
)
from app.db.mongo import get_db
from app.core.config import settings


router = APIRouter(prefix="/api/projects", tags=["projects"])


def _json_safe(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _normalize(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    return doc


@router.get("", response_model=list[ProjectOut])
async def list_projects_route(
    request: Request,
    q: str | None = None,
    status: str | None = None,
    user=Depends(withGuard(feature="view_decision", orgRole="viewer")),
):
    projects = await list_projects(
        user.get("tenant_id"),
        search=q,
        status=status,
    )
    return [_normalize(doc) for doc in projects]


@router.post("", response_model=ProjectOut)
async def create_project_route(
    payload: ProjectCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    project = await create_project(request.state.tenant_id, payload.model_dump())
    project_id = project.get("_id") or project.get("id")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.created",
        entity_type="project",
        entity_id=project_id,
    )
    return _normalize(project)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project_route(
    project_id: str,
    request: Request,
    _guard=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    project = await get_project(request.state.tenant_id, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return _normalize(project)


@router.get("/{project_id}/dashboard/owner-summary")
async def get_owner_dashboard_summary(
    project_id: str,
    request: Request,
    days: int = 7,
    _guard=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="days must be between 1 and 30")
    db = get_db()
    tenant_id = request.state.tenant_id
    tenant_oid = ObjectId(tenant_id)
    project_oid = ObjectId(project_id)
    now = datetime.now(timezone.utc)

    project = await db.projects.find_one(
        {"_id": project_oid, "tenant_id": tenant_oid},
        {"name": 1, "status": 1, "created_at": 1, "updated_at": 1, "description": 1},
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Count org users as the effective team size
    members_count = await db.users.count_documents(
        {"tenant_id": tenant_oid, "deleted_at": None, "is_active": True}
    )

    recent_activity_query = {
        "tenant_id": tenant_oid,
        "$or": [
            {"entity_id": project_id},
            {"metadata.project_id": project_id},
            {"action": {"$regex": "project", "$options": "i"}},
        ],
    }
    recent_activity_docs = await db.audit_logs.find(
        recent_activity_query,
        {"action": 1, "entity_type": 1, "entity_id": 1, "actor_id": 1, "created_at": 1, "metadata": 1},
    ).sort("_id", -1).limit(12).to_list(length=12)
    recent_activity = [
        {
            "id": str(doc["_id"]),
            "action": doc.get("action") or "event",
            "entity_type": doc.get("entity_type") or "unknown",
            "entity_id": _json_safe(doc.get("entity_id")) or "",
            "actor_id": str(doc.get("actor_id")) if doc.get("actor_id") else "",
            "created_at": doc.get("created_at"),
            "metadata": _json_safe(doc.get("metadata") or {}),
        }
        for doc in recent_activity_docs
    ]

    return {
        "window_days": days,
        "project": {
            "id": str(project["_id"]),
            "name": project.get("name") or "Project",
            "description": project.get("description") or "",
            "status": project.get("status") or "active",
            "created_at": project.get("created_at"),
            "updated_at": project.get("updated_at"),
        },
        "kpis": {
            "members": members_count,
            "decisions_total": 0,
            "decisions_window": 0,
            "prd_runs_window": 0,
            "active_prd_runs": 0,
        },
        "requirements": {
            "status": None,
            "created_at": None,
            "updated_at": None,
        },
        "prd": {
            "latest_status": None,
            "latest_created_at": None,
            "latest_updated_at": None,
            "latest_completed_at": None,
            "latest_version": None,
        },
        "llm_usage": {
            "window_days": days,
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "avg_tokens_per_request": 0.0,
            "max_tokens_per_request": 0,
            "token_budget_per_request": 0,
            "token_headroom_percent": 100.0,
            "by_feature": [],
            "daily": [],
        },
        "recent_decisions": [],
        "recent_activity": recent_activity,
    }


@router.put("/{project_id}", response_model=ProjectOut)
async def update_project_route(
    project_id: str,
    payload: ProjectUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    updated = await update_project(request.state.tenant_id, project_id, payload.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.updated",
        entity_type="project",
        entity_id=project_id,
    )
    return _normalize(updated)


@router.delete("/{project_id}")
async def delete_project_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    deleted = await delete_project(request.state.tenant_id, project_id, user.get("user_id"))
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.deleted",
        entity_type="project",
        entity_id=project_id,
    )
    return {"status": "deleted"}


@router.post("/{project_id}/restore")
async def restore_project_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    restored, reason = await restore_project(request.state.tenant_id, project_id)
    if not restored:
        if reason == "Project not found":
            raise HTTPException(status_code=404, detail=reason)
        raise HTTPException(status_code=400, detail=reason)
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="project.restored",
        entity_type="project",
        entity_id=project_id,
    )
    return {"status": "restored"}
