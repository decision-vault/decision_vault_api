from datetime import datetime, timezone
import httpx
import traceback
import logging
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks

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
from app.services.docs_management_service import DocsManagementService
from app.db.mongo import get_db
from app.core.config import settings

logger = logging.getLogger("decisionvault.projects")
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
    tenant_id = request.state.tenant_id
    db = get_db()
    
    # 1. Spawn parent database record tracking matrix fields
    project = await create_project(tenant_id, payload.model_dump())
    project_id = project.get("_id") or project.get("id")
    
    # 2. AUTOMATION HOOK: Instantly provision default workspace structure and a pristine baseline PRD anchor file
    try:
        await DocsManagementService.initialize_project_workspace(
            tenant_id=tenant_id,
            project_id=str(project_id),
            project_name=project.get("name", "New Project")
        )
    except Exception as hook_error:
        logger.error(f"Non-blocking workspace lifecycle initialization exception handled: {str(hook_error)}")

    # 3. AUTOMATION HOOK: Provision default Sprint Cycle
    try:
        from datetime import datetime, timedelta
        from app.services.task_service import SprintService
        await SprintService.create_sprint(
            tenant_id=tenant_id,
            payload={
                "project_id": str(project_id),
                "name": f"Sprint 1 - {project.get('name', 'New Project')}",
                "description": "Default initial sprint cycle.",
                "start_date": datetime.utcnow(),
                "end_date": datetime.utcnow() + timedelta(days=14)
            }
        )
    except Exception as sprint_error:
        logger.error(f"Non-blocking sprint lifecycle initialization exception handled: {str(sprint_error)}")



    # 5. AUTOMATION HOOK: Provision default Agentic Lifecycle Workflow
    try:
        from datetime import datetime
        default_nodes = [
            {
                "id": "node_init",
                "type": "input",
                "data": {"label": "🔒 Pipeline Initiator"},
                "position": {"x": 380, "y": 20},
                "style": {
                    "background": "var(--indigo-2)",
                    "color": "var(--indigo-11)",
                    "border": "1px dashed var(--indigo-6)",
                    "borderRadius": "12px",
                    "padding": "14px",
                    "width": "220px",
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "textAlign": "center",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.02)"
                }
            },
            {
                "id": "node_core",
                "data": {
                    "label": "🔑 Core Feature Module",
                    "summary": "Implements core feature specifications, data schema mapping, validation routes, and UI components.",
                    "agents": [
                        {
                            "role": "Designer Agent",
                            "status": "completed",
                            "percentage": 100,
                            "task": "Export typography tokens, figma screen layouts wireframes, validation feedback structures, and system token definitions.",
                            "logs": [
                                "Parsing Figma design template components...",
                                "Exporting styling system theme tokens...",
                                "Artifact published to assets database grid safely."
                            ]
                        },
                        {
                            "role": "Frontend Agent",
                            "status": "processing",
                            "percentage": 50,
                            "task": "Compile reactive validation schemas hooks with standard client boundaries.",
                            "logs": [
                                "Binding runtime form state hooks and change event listeners...",
                                "Injecting authentication validation layers constraints arrays..."
                            ]
                        },
                        {
                            "role": "Backend Agent",
                            "status": "processing",
                            "percentage": 40,
                            "task": "Expose security verification algorithms and token generation pipelines inside isolated FastAPI router modules.",
                            "logs": [
                                "Initializing security verification algorithms matrix parameters...",
                                "Hashing sample transaction tokens strings..."
                            ]
                        }
                    ]
                },
                "position": {"x": 365, "y": 160},
                "style": {
                    "background": "var(--color-surface)",
                    "color": "var(--gray-12)",
                    "border": "1px solid var(--gray-4)",
                    "borderRadius": "14px",
                    "padding": "18px",
                    "width": "250px",
                    "boxShadow": "0 10px 30px rgba(0,0,0,0.04)",
                    "cursor": "pointer"
                }
            }
        ]
        default_edges = [
            {
                "id": "e_init_core",
                "source": "node_init",
                "target": "node_core",
                "animated": True,
                "style": {"stroke": "var(--indigo-7)", "strokeWidth": 1.5}
            }
        ]
        await db.workflows.insert_one({
            "project_id": str(project_id),
            "tenant_id": tenant_id,
            "nodes": default_nodes,
            "edges": default_edges,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
    except Exception as workflow_error:
        logger.error(f"Non-blocking workflow lifecycle initialization exception handled: {str(workflow_error)}")

    # 6. Log mutation audit metrics rails details securely
    await log_event(
        tenant_id=tenant_id,
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

    project = await db.projects.find_one(
        {"_id": project_oid, "tenant_id": tenant_oid},
        {"name": 1, "status": 1, "created_at": 1, "updated_at": 1, "description": 1},
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

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
    background_tasks: BackgroundTasks, #  Injects FastAPI's asynchronous background task worker threads
    user=Depends(withGuard(feature="edit_decision", projectRole="project_admin")),
):
    tenant_id = request.state.tenant_id

    # 1. Schedule the deep document/chat history prune asynchronously out-of-thread
    background_tasks.add_task(
        DocsManagementService.delete_workspace_async_worker,
        tenant_id,
        project_id
    )
    logger.info(f"🚀 Scheduled async document cascade task for project_id: {project_id}")

    # 2. Complete the primary project container deletion immediately
    deleted = await delete_project(tenant_id, project_id, user.get("user_id"))
    if not deleted:
        raise HTTPException(status_code=404, detail="Target project not found")
        
    # 3. Log the mutation audit trail
    await log_event(
        tenant_id=tenant_id,
        actor_id=user.get("user_id"),
        action="project.deleted",
        entity_type="project",
        entity_id=project_id,
    )
    
    # Returns immediately—no more waiting for extensive NoSQL data prunes to finish computing
    return {
        "status": "deleted", 
        "message": "Project container removed. Workspace document cleanup processing in background."
    }

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


# =====================================================
# AI Workflow Generator Endpoints
# =====================================================
from pydantic import BaseModel
from typing import Optional
from app.services.workflow_generator_service import WorkflowGeneratorService

class TaskUpdatePayload(BaseModel):
    status: Optional[str] = None
    assigned_human_id: Optional[str] = None

@router.post("/{project_id}/workflow/generate")
async def generate_workflow_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    tenant_id = request.state.tenant_id
    workflow_id = await WorkflowGeneratorService.generate_workflow(tenant_id, project_id)
    
    await log_event(
        tenant_id=tenant_id,
        actor_id=user.get("user_id"),
        action="workflow.generated",
        entity_type="project",
        entity_id=project_id,
    )
    return {"workflow_id": workflow_id}

@router.get("/{project_id}/workflow")
async def get_workflow_route(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    tenant_id = request.state.tenant_id
    workflow = await WorkflowGeneratorService.get_project_workflow(tenant_id, project_id)
    return workflow

@router.patch("/{project_id}/workflow/tasks/{task_id}")
async def update_workflow_task_route(
    project_id: str,
    task_id: str,
    payload: TaskUpdatePayload,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    db = get_db()
    
    update_data = {}
    if payload.status is not None:
        update_data["status"] = payload.status
    if payload.assigned_human_id is not None:
        update_data["assigned_human_id"] = payload.assigned_human_id
        
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
        
    update_data["updated_at"] = datetime.utcnow()
    
    result = await db.workflow_tasks.find_one_and_update(
        {"_id": ObjectId(task_id)},
        {"$set": update_data},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
        
    # Recalculate and update workflow progress percentage
    workflow_id = result["workflow_id"]
    total_tasks = await db.workflow_tasks.count_documents({"workflow_id": workflow_id})
    completed_tasks = await db.workflow_tasks.count_documents({"workflow_id": workflow_id, "status": "completed"})
    blocked_tasks = await db.workflow_tasks.count_documents({"workflow_id": workflow_id, "status": "blocked"})
    
    progress = (completed_tasks / total_tasks * 100.0) if total_tasks > 0 else 0.0
    
    await db.project_workflows.find_one_and_update(
        {"_id": workflow_id},
        {
            "$set": {
                "statistics.completed_tasks": completed_tasks,
                "statistics.blocked_tasks": blocked_tasks,
                "statistics.progress": progress,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    return {"success": True}