from datetime import datetime
from pydantic import BaseModel, Field


class License(BaseModel):
    tenant_id: str = Field(..., description="Tenant ObjectId")
    plan: str = Field(..., description="trial | starter | team | enterprise")
    status: str = Field(..., description="trial | active | grace | expired | suspended")
    start_date: datetime
    expiry_date: datetime
    grace_period_days: int = Field(..., description="Grace period days after expiry")
    grace_start_date: datetime | None = None
    deleted_at: datetime | None = None
    deleted_by: str | None = Field(default=None, description="User ObjectId")


class LicenseCreate(BaseModel):
    plan: str = Field(..., description="trial | starter | team | enterprise")
    status: str = Field(default="active", description="trial | active | grace | expired | suspended")
    start_date: datetime
    expiry_date: datetime
    grace_period_days: int = 7
    grace_start_date: datetime | None = None


class LicenseUpdate(BaseModel):
    plan: str | None = Field(default=None, description="trial | starter | team | enterprise")
    status: str | None = Field(default=None, description="trial | active | grace | expired | suspended")
    start_date: datetime | None = None
    expiry_date: datetime | None = None
    grace_period_days: int | None = None
    grace_start_date: datetime | None = None
