from typing import Callable

from bson import ObjectId
from fastapi import Depends, HTTPException, Request

from app.core.rbac import (
    is_super_admin,
    org_permission_allows,
    org_role_at_least,
    project_permission_allows,
    project_role_at_least,
)
from app.db.mongo import get_db
from app.middleware.auth import get_current_user


def _forbidden(detail: str = "Forbidden") -> HTTPException:
    return HTTPException(status_code=403, detail=detail)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _resolve_tenant_id(request: Request) -> str | None:
    if hasattr(request.state, "tenant_id") and request.state.tenant_id:
        return request.state.tenant_id
    if "tenant_id" in request.path_params:
        return request.path_params["tenant_id"]
    if "tenant_id" in request.query_params:
        return request.query_params["tenant_id"]
    if request.headers.get("x-tenant-id"):
        return request.headers.get("x-tenant-id")
    return None


def _resolve_project_id(request: Request) -> str | None:
    if "project_id" in request.path_params:
        return request.path_params["project_id"]
    if "project_id" in request.query_params:
        return request.query_params["project_id"]
    if request.headers.get("x-project-id"):
        return request.headers.get("x-project-id")
    return None


def requireOrgRole(
    *,
    min_role: str | None = None,
    permission: str | None = None,
) -> Callable:
    async def _dependency(
        request: Request, user=Depends(get_current_user)
    ) -> dict:
        if is_super_admin(user.get("role")):
            return user

        tenant_id = _resolve_tenant_id(request) or user.get("tenant_id")
        if tenant_id != user.get("tenant_id"):
            raise _forbidden("Tenant mismatch")

        role = user.get("role")
        if permission and not org_permission_allows(role, permission):
            raise _forbidden("Insufficient org permission")
        if min_role and not org_role_at_least(role, min_role):
            raise _forbidden("Insufficient org role")

        return user

    return _dependency


def requireProjectRole(
    *,
    min_role: str | None = None,
    permission: str | None = None,
) -> Callable:
    async def _dependency(
        request: Request, user=Depends(get_current_user)
    ) -> dict:
        if is_super_admin(user.get("role")):
            return user

        tenant_id = _resolve_tenant_id(request) or user.get("tenant_id")
        if tenant_id != user.get("tenant_id"):
            raise _forbidden("Tenant mismatch")

        project_id = _resolve_project_id(request)
        if not project_id:
            raise _forbidden("Project id required")

        db = get_db()
        membership = await db.project_members.find_one(
            {
                "tenant_id": _oid(tenant_id),
                "project_id": _oid(project_id),
                "user_id": _oid(user.get("user_id")),
                "deleted_at": None,
            }
        )
        if not membership:
            raise _forbidden("Not a project member")

        role = membership.get("role")
        if permission and not project_permission_allows(role, permission):
            raise _forbidden("Insufficient project permission")
        if min_role and not project_role_at_least(role, min_role):
            raise _forbidden("Insufficient project role")

        request.state.project_role = role
        return user

    return _dependency
