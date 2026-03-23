from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class DemoRequestCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    company: str = Field(min_length=1, max_length=160)
    role: str | None = Field(default=None, max_length=120)
    team_size: int | None = Field(default=None, ge=1, le=100000)
    notes: str | None = Field(default=None, max_length=4000)
    preferred_time: str | None = Field(default=None, max_length=200)
    timezone: str | None = Field(default=None, max_length=80)

    # Honeypot field: real users won't fill this.
    website: str | None = Field(default=None, max_length=200)


class DemoRequestResponse(BaseModel):
    status: str = "ok"
    request_id: str
    created_at: datetime

