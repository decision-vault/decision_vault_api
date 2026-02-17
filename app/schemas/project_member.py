from datetime import datetime
from pydantic import BaseModel, Field


class ProjectMember(BaseModel):
    tenant_id: str = Field(..., description="Tenant ObjectId")
    project_id: str = Field(..., description="Project ObjectId")
    user_id: str = Field(..., description="User ObjectId")
    role: str = Field(..., description="project_admin | contributor | viewer")
    created_at: datetime
    deleted_at: datetime | None = None
    deleted_by: str | None = Field(default=None, description="User ObjectId")


class ProjectMemberCreate(BaseModel):
    user_id: str = Field(..., description="User ObjectId")
    role: str = Field(..., description="project_admin | contributor | viewer")


class ProjectMemberUpdate(BaseModel):
    role: str = Field(..., description="project_admin | contributor | viewer")
