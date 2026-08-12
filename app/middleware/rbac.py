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
from app.services.tenant_service import user_owns_tenant


def _forbidden(detail: str = "Forbidden") -> HTTPException:
    return HTTPException(status_code=403, detail=detail)

def _bad_request(detail: str = "Bad request") -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _oid(value: str) -> ObjectId:
    if not value or not ObjectId.is_valid(value):
        raise _bad_request("Invalid id")
    return ObjectId(value)


async def _resolve_effective_role(request: Request, user: dict, tenant_id: str) -> tuple[str, str]:
    """Resolve (tenant_id, role) allowing orgs the user owns.

    Returns the primary tenant role normally, or ``owner`` for owned
    organizations outside the token's primary tenant.
    """
    if tenant_id != user.get("tenant_id"):
        owned = await user_owns_tenant(user.get("user_id"), tenant_id)
        if not owned:
            raise _forbidden("Tenant mismatch")
        return tenant_id, "owner"
    return tenant_id, user.get("role")


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
        tenant_id, role = await _resolve_effective_role(request, user, tenant_id)
        request.state.tenant_role = role

        if permission and not org_permission_allows(role, permission):
            raise _forbidden("Insufficient org permission")
        if min_role and not org_role_at_least(role, min_role):
            raise _forbidden("Insufficient org role")

        return user

    return _dependency


def _derive_project_role(org_role: str | None) -> str:
    """Map an org role to a default project role.

    Org owners and admins get ``project_admin``; everyone else gets
    ``contributor`` so they can still read and write project data.
    """
    if (org_role or "").lower() in {"owner", "admin"}:
        return "project_admin"
    return "contributor"


async def _resolve_project_role(db, project: dict, user: dict) -> str | None:
    """Resolve a user's role within a project.

    Priority: explicit ``project_members`` membership > implicit project owner
    (``owner_id``) > legacy fallback derived from the org role.
    """
    user_id = user.get("user_id")
    if not user_id:
        return None

    if project.get("owner_id") and str(project["owner_id"]) == user_id:
        return "project_admin"

    membership = await db.project_members.find_one(
        {
            "project_id": project["_id"],
            "user_id": _oid(user_id),
            "removed_at": None,
        }
    )
    if membership:
        return membership.get("role") or "contributor"

    return _derive_project_role(user.get("role"))


def requireProjectRole(
    *,
    min_role: str | None = None,
    permission: str | None = None,
) -> Callable:
    async def _dependency(
        request: Request, user=Depends(get_current_user)
    ) -> dict:
        if is_super_admin(user.get("role")):
            request.state.project_role = "project_admin"
            return user

        tenant_id = _resolve_tenant_id(request) or user.get("tenant_id")
        tenant_id, role = await _resolve_effective_role(request, user, tenant_id)

        project_id = _resolve_project_id(request)
        if not project_id:
            raise _forbidden("Project id required")

        # Verify the project belongs to the user's tenant
        db = get_db()
        project = await db.projects.find_one(
            {
                "_id": _oid(project_id),
                "tenant_id": _oid(tenant_id),
                "deleted_at": None,
            }
        )
        if not project:
            raise _forbidden("Project not found in tenant")

        role = await _resolve_project_role(db, project, user)
        if role is None:
            raise _forbidden("You are not a member of this project")
        if permission and not project_permission_allows(role, permission):
            raise _forbidden("Insufficient project permission")
        if min_role and not project_role_at_least(role, min_role):
            raise _forbidden("Insufficient project role")

        request.state.project_role = role
        return user

    return _dependency
