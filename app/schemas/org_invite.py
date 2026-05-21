from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class OrgInviteProjectAccess(BaseModel):
    project_id: str = Field(..., min_length=12, max_length=64)
    project_role: str = Field(default="contributor", min_length=2, max_length=32)


class OrgInviteCreate(BaseModel):
    email: EmailStr
    role: str = Field(default="member", min_length=2, max_length=32)
    project_access: list[OrgInviteProjectAccess] | None = None


class OrgInviteOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    status: str
    created_at: datetime
    expires_at: datetime
    accepted_at: datetime | None = None
    revoked_at: datetime | None = None


class OrgInviteCreateResponse(BaseModel):
    invite: OrgInviteOut
    invite_link: str


class OrgInviteAccept(BaseModel):
    token: str = Field(..., min_length=16, max_length=512)
    password: str | None = Field(default=None, min_length=8, max_length=128)
