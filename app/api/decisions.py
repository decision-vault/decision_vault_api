from fastapi import APIRouter, Depends

from app.middleware.guard import withGuard


router = APIRouter(prefix="/api", tags=["decisions"])


@router.post("/projects/{project_id}/decisions")
async def create_decision(
    project_id: str,
    _guard=Depends(withGuard(feature="create_decision", projectRole="contributor")),
):
    return {"status": "decision created", "project_id": project_id}


@router.put("/decisions/{decision_id}")
async def edit_decision(
    decision_id: str,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    return {"status": "decision updated", "decision_id": decision_id}


@router.post("/decisions/{decision_id}/relationships")
async def add_relationship(
    decision_id: str,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    return {"status": "relationship added", "decision_id": decision_id}
