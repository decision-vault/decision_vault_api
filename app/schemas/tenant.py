from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    plan: str = Field(default="free", max_length=20)


class TenantUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)


class TenantOut(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    plan: Optional[str] = None
    plan_status: Optional[str] = None
