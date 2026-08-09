from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
import httpx
from datetime import datetime
from bson import ObjectId

from app.middleware.guard import withGuard
from app.db.mongo import get_db
from app.core.config import settings
from app.services.docs_management_service import DocsManagementService
from app.services.workflow_service import WorkflowService
from app.services.task_service import TaskService, SprintService
from app.utils.serialize import serialize_doc

router = APIRouter(prefix="/api/prd-planner", tags=["AI PRD Planner"])

class GeneratePlanRequest(BaseModel):
    project_id: str
    document_id: str

@router.post("/generate-plan")
async def generate_plan_endpoint(
    payload: GeneratePlanRequest,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    tenant_id = request.state.tenant_id
    db = get_db()
    
    # 1. Fetch document from MongoDB
    doc = await DocsManagementService.get_document_by_id(payload.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    prd_body = doc.get("body", "")
    
    # 2. Fetch project details
    project = await db.projects.find_one({"_id": ObjectId(payload.project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    product_name = project.get("name", "New Project")
    
    # 3. Request LangGraph service to generate plan
    langgraph_url = f"{settings.langgraph_url}/workflow/plan/generate"
    micro_payload = {
        "tenant_id": tenant_id,
        "project_id": payload.project_id,
        "document_id": payload.document_id,
        "product_name": product_name,
        "prd_body": prd_body
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(langgraph_url, json=micro_payload, timeout=None)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Planner agent failed: {response.text}")
            
            data = response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to communicate with planner agent: {str(e)}")
            
    # 4. Save workflow layout details (nodes and edges)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    await WorkflowService.update_workflow(payload.project_id, {
        "nodes": nodes,
        "edges": edges
    })
    
    # 5. Fetch default sprint for project_id
    sprint = await db.sprints.find_one({"project_id": ObjectId(payload.project_id), "deleted_at": None})
    sprint_id = str(sprint["_id"]) if sprint else None
    
    # 6. Create tasks in the project
    tasks = data.get("tasks", [])
    for task_item in tasks:
        try:
            task_payload = {
                "project_id": payload.project_id,
                "sprint_id": sprint_id,
                "title": task_item.get("title", "Generated Task"),
                "description": task_item.get("description", ""),
                "role": task_item.get("role", "backend_developer"),
                "priority": task_item.get("priority", "medium"),
                "story_points": task_item.get("story_points", 3)
            }
            await TaskService.create_task(tenant_id=tenant_id, user_id=user.get("user_id"), payload=task_payload)
        except Exception as t_err:
            pass
            
    # 7. Append plan card message to chat history
    now = datetime.utcnow()
    plan_snapshot = {
        "timestamp": now.isoformat(),
        "agent_prompt_or_chat": "System Plan Generated",
        "saved_snapshot_body": "🎉 Project Plan Generated! Click below to open the interactive workflow board.",
        "is_plan_card": True
    }
    
    await db.documents.find_one_and_update(
        {"_id": ObjectId(payload.document_id)},
        {"$push": {"chat_history": plan_snapshot}}
    )
    
    return {
        "success": True,
        "nodes_count": len(nodes),
        "tasks_count": len(tasks)
    }
