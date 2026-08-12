from fastapi import Depends, HTTPException, Request

from app.db.mongo import get_db
from app.middleware.auth import get_current_user
from app.services.tenant_service import user_owns_tenant


def _forbidden(detail: str = "Forbidden") -> HTTPException:
    return HTTPException(status_code=403, detail=detail)


def _resolve_tenant_id(request: Request) -> str | None:
    if "tenant_id" in request.path_params:
        return request.path_params["tenant_id"]
    if "org_id" in request.path_params:
        return request.path_params["org_id"]
    if "tenant_id" in request.query_params:
        return request.query_params["tenant_id"]
    if request.headers.get("x-tenant-id"):
        return request.headers.get("x-tenant-id")
    return None


async def resolve_tenant(request: Request, user=Depends(get_current_user)) -> str:
    tenant_id = _resolve_tenant_id(request) or user.get("tenant_id")
    if tenant_id != user.get("tenant_id"):
        # Allow access to organizations the user owns, even when it is not
        # the primary tenant in their token (multi-org ownership).
        owned = await user_owns_tenant(user.get("user_id"), tenant_id)
        if not owned:
            raise _forbidden("Tenant mismatch")
        request.state.tenant_role = "owner"
    request.state.tenant_id = tenant_id
    return tenant_id
