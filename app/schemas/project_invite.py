from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class ProjectInviteCreate(BaseModel):
    email: EmailStr
    role: str = Field(default="contributor", min_length=2, max_length=32)


class ProjectInviteOut(BaseModel):
    id: str
    project_id: str
    email: EmailStr
    role: str
    status: str
    created_at: datetime
    expires_at: datetime
    accepted_at: datetime | None = None
    declined_at: datetime | None = None
    revoked_at: datetime | None = None


class ProjectInviteCreateResponse(BaseModel):
    invite: ProjectInviteOut
    invite_link: str


class ProjectInviteToken(BaseModel):
    token: str = Field(..., min_length=16, max_length=512)


class ProjectMemberOut(BaseModel):
    user_id: str
    email: str
    name: str | None = None
    role: str
    is_owner: bool = False
    joined_at: datetime | None = None


class ProjectMemberRoleUpdate(BaseModel):
    role: str = Field(..., min_length=2, max_length=32)


class ProjectInviteAccepted(BaseModel):
    tenant_id: str
    project_id: str
    project_name: str
    role: str
