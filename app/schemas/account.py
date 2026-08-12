from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class AccountOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    provider: str = ""
    is_active: bool = True
    full_name: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    tenant_id: str = ""
    tenant_name: str = ""
    tenant_slug: str = ""


class AccountUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=80)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)
