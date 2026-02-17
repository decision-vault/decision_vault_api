from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=160)
    description: Optional[str] = Field(default=None, max_length=2000)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=160)
    description: Optional[str] = Field(default=None, max_length=2000)


class ProjectOut(BaseModel):
    id: str
    tenant_id: str
    name: str
    slug: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
