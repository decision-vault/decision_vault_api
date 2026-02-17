from typing import Callable

from fastapi import Depends, Request

from app.middleware.auth import get_current_user
from app.middleware.license import assertLicense
from app.middleware.rbac import requireOrgRole, requireProjectRole
from app.middleware.tenant import resolve_tenant


def withGuard(
    *,
    feature: str,
    orgRole: str | None = None,
    projectRole: str | None = None,
) -> Callable:
    async def _dependency(
        request: Request,
        user=Depends(get_current_user),
        _tenant=Depends(resolve_tenant),
        _license=Depends(assertLicense(feature)),
    ):
        if orgRole:
            await requireOrgRole(min_role=orgRole)(request, user)
        if projectRole:
            await requireProjectRole(min_role=projectRole)(request, user)
        return user

    return _dependency
