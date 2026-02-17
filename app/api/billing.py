from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.middleware.guard import withGuard
from app.services.billing_service import create_checkout_session


router = APIRouter(prefix="/api", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan: str = Field(..., description="starter | team | enterprise")
    success_url: str
    cancel_url: str


@router.post("/orgs/{org_id}/billing/checkout")
async def billing_checkout(
    org_id: str,
    payload: CheckoutRequest,
    _guard=Depends(withGuard(feature="view_decision", orgRole="owner")),
):
    try:
        return await create_checkout_session(org_id, payload.plan, payload.success_url, payload.cancel_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
