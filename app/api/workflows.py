from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowResponse
from app.services.workflow_service import WorkflowService
from app.middleware.guard import withGuard
from app.utils.serialize import serialize_doc

router = APIRouter(prefix="/api/workflows", tags=["Agentic Lifecycle Builder"])

@router.post("", response_model=WorkflowResponse)
async def create_workflow(
    payload: WorkflowCreate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    tenant_id = request.state.tenant_id
    existing = await WorkflowService.get_workflow_by_project(payload.project_id)
    if existing:
        raise HTTPException(status_code=400, detail="Workflow for this project already exists.")
    
    workflow = await WorkflowService.create_workflow(tenant_id=tenant_id, payload=payload.model_dump())
    serialized = serialize_doc(workflow)
    serialized["id"] = str(serialized["_id"])
    return serialized

@router.get("", response_model=WorkflowResponse)
async def get_workflow(
    project_id: str,
    request: Request,
    user=Depends(withGuard(feature="view_decision", projectRole="viewer"))
):
    workflow = await WorkflowService.get_workflow_by_project(project_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    serialized = serialize_doc(workflow)
    serialized["id"] = str(serialized["_id"])
    return serialized

@router.put("", response_model=WorkflowResponse)
async def update_workflow(
    project_id: str,
    payload: WorkflowUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor"))
):
    workflow = await WorkflowService.update_workflow(project_id, payload.model_dump(exclude_none=True))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    serialized = serialize_doc(workflow)
    serialized["id"] = str(serialized["_id"])
    return serialized
