from datetime import datetime, timezone
from pydantic import BaseModel, Field, HttpUrl, field_validator


class CustomDecisionPayload(BaseModel):
    tenant_id: str
    project_id: str
    decision_title: str = Field(..., min_length=3, max_length=200)
    decision_statement: str = Field(..., min_length=3)
    context: str | None = None
    source_url: HttpUrl
    timestamp: datetime
    external_id: str = Field(..., min_length=3, max_length=200)

    @field_validator("timestamp", mode="before")
    @classmethod
    def normalize_timestamp(cls, value):
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return value

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone(cls, value: datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamp must include timezone offset")
        return value


class CustomDecisionResponse(BaseModel):
    status: str
    decision_id: str | None = None
    idempotent: bool = False


class OAuthClientCreate(BaseModel):
    name: str | None = None


class OAuthClientResponse(BaseModel):
    client_id: str
    client_secret: str | None = None
    name: str | None = None


class OAuthTokenRequest(BaseModel):
    grant_type: str = Field(..., pattern="^client_credentials$")
    client_id: str
    client_secret: str
    scope: str | None = None


class OAuthTokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    scope: str | None = None
