from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request

from app.middleware.guard import withGuard
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate
from app.services.audit_service import log_event
from app.services.project_member_service import add_project_member
from app.services.project_service import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    update_project,
)
from app.db.mongo import get_db


router = APIRouter(prefix="/api/projects", tags=["projects"])


def _normalize(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    return doc


@router.get("", response_model=list[ProjectOut])
async def list_projects_route(
    request: Request,
    user=Depends(withGuard(feature="view_decision", orgRole="viewer")),
):
    db = get_db()
    memberships = await db.project_members.find(
        {
            "tenant_id": ObjectId(user.get("tenant_id")),
            "user_id": ObjectId(user.get("user_id")),
            "deleted_at": None,
        }
    ).to_list(length=500)
    project_ids = [str(doc["project_id"]) for doc in memberships]
    if not project_ids:
        return []
    projects = await list_projects(user.get("tenant_id"), project_ids=project_ids)
    return [_normalize(doc) for doc in projects]


@router.post("", response_model=ProjectOut)
async def create_project_route(
    payload: ProjectCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    project = await create_project(request.state.tenant_id, payload.model_dump())
    project_id = project.get("_id") or project.get("id")
    await add_project_member(request.state.tenant_id, project_id, user.get("user_id"), "project_admin")
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
