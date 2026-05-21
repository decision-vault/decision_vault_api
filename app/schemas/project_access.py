from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


PROJECT_ROLES = {"viewer", "contributor", "project_admin"}


class ProjectCatalogOut(BaseModel):
    id: str
    name: str
    status: str | None = None


class ProjectAccessRequestOut(BaseModel):
    id: str
    project_id: str
    project_name: str | None = None
    user_id: str
    user_email: EmailStr
    status: str
    created_at: datetime
    decided_at: datetime | None = None
    decided_by_user_id: str | None = None


class ProjectInviteByEmail(BaseModel):
    email: EmailStr
    role: str = Field(default="contributor", min_length=2, max_length=32)

