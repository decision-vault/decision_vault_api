from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class OrgUserOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    provider: str = ""
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class OrgUserUpdate(BaseModel):
    is_active: bool = Field(...)

