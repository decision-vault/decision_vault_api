from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.rbac import is_super_admin
from app.middleware.auth import get_current_user
from app.middleware.guard import withGuard
from app.schemas.tenant import TenantCreate, TenantOut, TenantUpdate
from app.services.audit_service import log_event
from app.services.tenant_service import create_tenant, delete_tenant, get_tenant, list_tenants, update_tenant


router = APIRouter(prefix="/api/orgs", tags=["orgs"])


def _normalize(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    return doc


@router.get("/me", response_model=TenantOut)
async def get_org(
    request: Request,
    user=Depends(withGuard(feature="view_decision", orgRole="viewer")),
):
    tenant = await get_tenant(request.state.tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Organization not found")
    return _normalize(tenant)


@router.patch("/me", response_model=TenantOut)
async def update_org(
    payload: TenantUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    updated = await update_tenant(request.state.tenant_id, payload.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Organization not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.updated",
        entity_type="tenant",
        entity_id=request.state.tenant_id,
    )
    return _normalize(updated)


@router.delete("/me")
async def delete_org(
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    deleted = await delete_tenant(request.state.tenant_id, user.get("user_id"))
    if not deleted:
        raise HTTPException(status_code=404, detail="Organization not found")
    await log_event(
        tenant_id=request.state.tenant_id,
        actor_id=user.get("user_id"),
        action="org.deleted",
        entity_type="tenant",
        entity_id=request.state.tenant_id,
    )
    return {"status": "deleted"}


@router.get("", response_model=list[TenantOut])
async def list_orgs(user=Depends(get_current_user)):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Forbidden")
    tenants = await list_tenants()
    return [_normalize(doc) for doc in tenants]


@router.post("", response_model=TenantOut)
async def create_org(payload: TenantCreate, user=Depends(get_current_user)):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Forbidden")
    tenant = await create_tenant(payload.name)
    return _normalize(tenant)
