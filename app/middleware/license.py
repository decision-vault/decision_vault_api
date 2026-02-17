from typing import Callable

from fastapi import Depends, Request

from app.core.license import FEATURE_REQUIREMENTS, WRITE_BLOCKED_FEATURES
from app.core.errors import LicenseError
from app.middleware.auth import get_current_user
from app.services.license_service import evaluate_license_status, get_current_license


def assertLicense(feature: str) -> Callable:
    async def _dependency(request: Request, user=Depends(get_current_user)) -> dict:
        tenant_id = user.get("tenant_id")
        license_doc = await get_current_license(tenant_id)
        if not license_doc:
            raise LicenseError("LICENSE_MISSING", "License missing")

        status_info = evaluate_license_status(license_doc)
        if status_info["status"] == "suspended":
            raise LicenseError("LICENSE_SUSPENDED", "License suspended")

        feature_rule = FEATURE_REQUIREMENTS.get(feature, {})
        if feature_rule.get("always"):
            request.state.license = license_doc
            request.state.license_status = status_info
            return user

        if status_info["status"] in {"expired", "grace"} and feature in WRITE_BLOCKED_FEATURES:
            raise LicenseError("LICENSE_EXPIRED", "License expired (read-only)")

        if status_info["status"] in {"expired", "grace"}:
            request.state.license = license_doc
            request.state.license_status = status_info
            return user

        allowed_plans = feature_rule.get("plans")
        if allowed_plans and license_doc.get("plan") not in allowed_plans:
            raise LicenseError("LICENSE_INSUFFICIENT", "License plan insufficient")

        request.state.license = license_doc
        request.state.license_status = status_info
        return user

    return _dependency
