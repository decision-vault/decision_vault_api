from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.schemas.task import (
    TaskCreate, TaskUpdate, TaskStatusUpdate, TaskAssign, CommentCreate,
    SprintCreate, SprintUpdate
)
from app.services.task_service import TaskService, SprintService
from app.middleware.tenant import resolve_tenant
from app.middleware.auth import get_current_user
from app.utils.serialize import serialize_doc

router = APIRouter(prefix="/api", tags=["Project Management"])

"""
=========================
SPRINT CRUD
=========================
"""

@router.post("/sprints")
async def create_sprint(payload: SprintCreate, tenant_id: str = Depends(resolve_tenant)):
    sprint = await SprintService.create_sprint(tenant_id=tenant_id, payload=payload.model_dump())
    return {"id": str(sprint["_id"])}

@router.get("/sprints/{sprint_id}")
async def get_sprint(sprint_id: str):
    sprint = await SprintService.get_sprint(sprint_id)
    if not sprint:
        raise HTTPException(status_code=404, detail="Sprint not found")
    sprint = serialize_doc(sprint)
    sprint["id"] = str(sprint["_id"])
    return sprint

@router.put("/sprints/{sprint_id}")
async def update_sprint(sprint_id: str, payload: SprintUpdate):
    sprint = await SprintService.update_sprint(sprint_id, payload.model_dump(exclude_none=True))
    if not sprint:
        raise HTTPException(status_code=404, detail="Sprint not found")
    sprint = serialize_doc(sprint)
    sprint["id"] = str(sprint["_id"])
    return sprint

@router.delete("/sprints/{sprint_id}")
async def delete_sprint(sprint_id: str):
    await SprintService.delete_sprint(sprint_id)
    return {"success": True}

@router.get("/sprints")
async def list_sprints(project_id: str, tenant_id: str = Depends(resolve_tenant)):
    """
    Fetches all sprint cycles associated with a specific project id.
    """
    sprints = await SprintService.list_sprints(tenant_id=tenant_id, project_id=project_id)
    return [{**serialize_doc(s), "id": str(s["_id"])} for s in sprints]


"""
=========================
TASK CRUD
=========================
"""

@router.post("/tasks")
async def create_task(
    payload: TaskCreate,
    tenant_id: str = Depends(resolve_tenant),
    user: dict = Depends(get_current_user),
):
    user_id = user.get("user_id") if user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    task = await TaskService.create_task(
        tenant_id=tenant_id, user_id=user_id, payload=payload.model_dump()
    )
    return {"id": str(task["_id"])}

@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = await TaskService.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task = serialize_doc(task)
    task["id"] = str(task["_id"])
    return task

@router.get("/tasks")
async def list_tasks(project_id: str):
    tasks = await TaskService.list_tasks(project_id)
    return [{**serialize_doc(t), "id": str(t["_id"])} for t in tasks]

@router.put("/tasks/{task_id}")
async def update_task(task_id: str, payload: TaskUpdate):
    task = await TaskService.update_task(task_id, payload.model_dump(exclude_none=True))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task = serialize_doc(task)
    task["id"] = str(task["_id"])
    return task

@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    await TaskService.delete_task(task_id)
    return {"success": True}

@router.patch("/tasks/{task_id}/status")
async def update_status(task_id: str, payload: TaskStatusUpdate):
    task = await TaskService.update_task(task_id, {"status": payload.status})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task = serialize_doc(task)
    task["id"] = str(task["_id"])
    return task


"""
=========================
 SUBTASK CRUD ENDPOINTS
=========================
"""

@router.post("/tasks/{parent_id}/subtasks")
async def create_subtask(
    parent_id: str,
    payload: TaskCreate,
    tenant_id: str = Depends(resolve_tenant),
    user: dict = Depends(get_current_user),
):
    """
    Creates a subtask nested under an explicit parent task context route parameter.
    """
    user_id = user.get("user_id") if user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Enforce parent_id parity inside the data payload mapping
    data = payload.model_dump()
    data["parent_id"] = parent_id

    subtask = await TaskService.create_task(tenant_id=tenant_id, user_id=user_id, payload=data)
    return {"id": str(subtask["_id"])}

@router.get("/tasks/{parent_id}/subtasks")
async def get_subtasks(parent_id: str):
    """
    Lists all active child subtasks nested under a parent task.
    """
    subtasks = await TaskService.list_subtasks(parent_id)
    return [{**serialize_doc(st), "id": str(st["_id"])} for st in subtasks]

@router.get("/subtasks/{subtask_id}")
async def get_subtask_detail(subtask_id: str):
    """
    Fetches details of a specific individual subtask.
    """
    subtask = await TaskService.get_task(subtask_id)
    if not subtask or not subtask.get("parent_id"):
        raise HTTPException(status_code=404, detail="Subtask not found")
    subtask = serialize_doc(subtask)
    subtask["id"] = str(subtask["_id"])
    return subtask

@router.put("/subtasks/{subtask_id}")
async def update_subtask_detail(subtask_id: str, payload: TaskUpdate):
    """
    Updates attributes of a specific subtask inline.
    """
    subtask = await TaskService.get_task(subtask_id)
    if not subtask or not subtask.get("parent_id"):
        raise HTTPException(status_code=404, detail="Subtask not found")
        
    updated = await TaskService.update_task(subtask_id, payload.model_dump(exclude_none=True))
    updated = serialize_doc(updated)
    updated["id"] = str(updated["_id"])
    return updated

@router.delete("/subtasks/{subtask_id}")
async def delete_subtask_detail(subtask_id: str):
    """
    Removes an isolated subtask record without affecting its siblings or parents.
    """
    subtask = await TaskService.get_task(subtask_id)
    if not subtask or not subtask.get("parent_id"):
        raise HTTPException(status_code=404, detail="Subtask not found")
        
    await TaskService.delete_task(subtask_id)
    return {"success": True}


"""
=========================
ASSIGNMENTS, COMMENTS & ACTIVITIES
=========================
"""

@router.post("/tasks/{task_id}/assign")
async def assign_task(task_id: str, payload: TaskAssign):
    task = await TaskService.assign_task(task_id, payload.user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task = serialize_doc(task)
    task["id"] = str(task["_id"])
    return task

@router.post("/tasks/{task_id}/unassign")
async def unassign_task(task_id: str):
    task = await TaskService.unassign_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task = serialize_doc(task)
    task["id"] = str(task["_id"])
    return task

@router.get("/tasks/{task_id}/comments")
async def get_task_comments(task_id: str):
    comments = await TaskService.get_task_comments(task_id)
    return [{**serialize_doc(c), "id": str(c["_id"])} for c in comments]

@router.post("/tasks/{task_id}/comments")
async def create_task_comment(task_id: str, payload: CommentCreate, user: dict = Depends(get_current_user)):
    user_id = user.get("user_id") if user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    comment = await TaskService.create_task_comment(task_id=task_id, user_id=user_id, message=payload.message)
    comment = serialize_doc(comment)
    comment["id"] = str(comment["_id"])
    return comment

@router.get("/tasks/{task_id}/activities")
async def get_task_activities(task_id: str):
    activities = await TaskService.get_task_activities(task_id)
    return [{**serialize_doc(a), "id": str(a["_id"])} for a in activities]

@router.get("/sprints/{sprint_id}/tasks")
async def get_tasks_by_sprint(sprint_id: str):
    tasks = await TaskService.list_tasks_by_sprint(sprint_id)
    return [{**serialize_doc(t), "id": str(t["_id"])} for t in tasks]