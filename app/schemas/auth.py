from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    tenant_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    tenant_id: str | None = None
    tenant_slug: str | None = None
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class GoogleAuthRequest(BaseModel):
    tenant_id: str | None = None
    tenant_slug: str | None = None
