from fastapi import APIRouter, Depends

from app.middleware.guard import withGuard
from app.middleware.rbac import requireProjectRole


router = APIRouter(prefix="/api", tags=["security"])


@router.post("/billing")
async def manage_billing(
    _guard=Depends(withGuard(feature="view_decision", orgRole="owner")),
):
    return {"status": "billing updated"}


@router.post("/integrations")
async def manage_integrations(
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    return {"status": "integration updated"}


@router.post("/decisions")
async def create_decision(
    _guard=Depends(withGuard(feature="create_decision", orgRole="member")),
):
    return {"status": "decision created"}


@router.put("/decisions/{decision_id}", operation_id="example_edit_decision")
async def edit_decision(
    decision_id: str,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    return {"status": "decision updated", "decision_id": decision_id}


@router.post("/slack/capture")
async def slack_capture(
    _guard=Depends(withGuard(feature="slack_capture", orgRole="admin")),
):
    return {"status": "slack capture queued"}


@router.get("/projects/{project_id}")
async def read_project(
    project_id: str,
    _guard=Depends(withGuard(feature="view_decision", projectRole="viewer")),
):
    return {"status": "project read", "project_id": project_id}
