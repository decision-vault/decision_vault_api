from fastapi import APIRouter, Depends, HTTPException, Request
from app.middleware.auth import get_current_user
from app.db.mongo import get_db
from bson import ObjectId

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

# Simple schema definitions
from pydantic import BaseModel

class OnboardingPayload(BaseModel):
    purpose: str | None = None
    tools: list[str] = []
    features: list[str] = []
    workspace_name: str | None = None
    source: str | None = None

# GET onboarding data for current tenant
@router.get("", response_model=OnboardingPayload)
async def get_onboarding(request: Request, user=Depends(get_current_user)):
    tenant_id = request.state.tenant_id
    db = get_db()
    doc = await db.onboarding.find_one({"tenant_id": ObjectId(tenant_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Onboarding data not found")
    return OnboardingPayload(**doc)

# POST onboarding data (create or update)
@router.post("", response_model=OnboardingPayload)
async def post_onboarding(payload: OnboardingPayload, request: Request, user=Depends(get_current_user)):
    tenant_id = request.state.tenant_id
    db = get_db()
    existing = await db.onboarding.find_one({"tenant_id": ObjectId(tenant_id)})
    if existing:
        await db.onboarding.update_one(
            {"tenant_id": ObjectId(tenant_id)},
            {"$set": payload.model_dump()},
        )
    else:
        await db.onboarding.insert_one({
            "tenant_id": ObjectId(tenant_id),
            **payload.model_dump(),
        })
    return payload
