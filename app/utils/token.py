import hashlib
import hmac
import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import settings


ALGORITHM = "HS256"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: str, tenant_id: str, role: str) -> tuple[str, int]:
    expires = _utcnow() + timedelta(minutes=settings.access_token_minutes)
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "type": "access",
        "exp": int(expires.timestamp()),
        "iat": int(_utcnow().timestamp()),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)
    return token, settings.access_token_minutes * 60


def create_refresh_token(user_id: str, tenant_id: str, role: str) -> tuple[str, str, datetime]:
    expires = _utcnow() + timedelta(days=settings.refresh_token_days)
    jti = str(uuid.uuid4())
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "type": "refresh",
        "jti": jti,
        "exp": int(expires.timestamp()),
        "iat": int(_utcnow().timestamp()),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)
    return token, jti, expires


def decode_token(token: str) -> dict:
    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[ALGORITHM],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
    )


def hash_token(token: str) -> str:
    secret = settings.jwt_secret.encode("utf-8")
    return hmac.new(secret, token.encode("utf-8"), hashlib.sha256).hexdigest()
