from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)


class TenantUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)


class TenantOut(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime
