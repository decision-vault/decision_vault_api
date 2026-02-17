from fastapi import APIRouter, Depends

from app.middleware.guard import withGuard


router = APIRouter(prefix="/api", tags=["uploads"])


@router.post("/projects/{project_id}/decisions/upload")
async def upload_decision_document(
    project_id: str,
    _guard=Depends(withGuard(feature="upload_document", projectRole="contributor")),
):
    return {"status": "upload queued", "project_id": project_id}
