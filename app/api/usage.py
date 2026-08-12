from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.middleware.guard import withGuard
from app.services.usage_service import get_usage_overview, record_usage

router = APIRouter(prefix="/api", tags=["usage"])


class UsageRecordIn(BaseModel):
    project_id: str | None = None
    feature: str = "generic"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@router.get("/orgs/me/usage")
async def get_usage(
    request: Request,
    period: str = "current",
    project_id: str | None = None,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    if period not in ("current", "previous"):
        raise HTTPException(status_code=400, detail="period must be 'current' or 'previous'")
    if project_id and not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="invalid project_id")
    return await get_usage_overview(
        request.state.tenant_id, period=period, project_id=project_id
    )


@router.post("/usage/record")
async def post_usage_record(
    payload: UsageRecordIn,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    await record_usage(
        request.state.tenant_id,
        project_id=payload.project_id,
        feature=payload.feature,
        prompt_tokens=payload.prompt_tokens,
        completion_tokens=payload.completion_tokens,
    )
    return {"ok": True}
