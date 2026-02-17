from typing import TypedDict


class LicenseContext(TypedDict):
    status: str
    read_only: bool
    days_remaining: int | None
    grace_remaining: int | None
    message: str | None


class RequestContext(TypedDict):
    tenant_id: str
    license: dict
    license_status: LicenseContext
