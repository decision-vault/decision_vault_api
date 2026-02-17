from bson import ObjectId
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db.mongo import get_db
from app.utils.token import decode_token


bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Unauthorized") -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


async def get_current_user(
    request: Request, creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    if not creds or creds.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        payload = decode_token(creds.credentials)
    except Exception:
        raise _unauthorized("Invalid token")

    if payload.get("type") != "access":
        raise _unauthorized("Invalid token type")

    db = get_db()
    user = await db.users.find_one({"_id": _oid(payload.get("sub"))})
    if not user:
        raise _unauthorized("User not found")

    if str(user.get("tenant_id")) != payload.get("tenant_id"):
        raise _unauthorized("Tenant mismatch")

    request.state.user = {
        "user_id": payload.get("sub"),
        "tenant_id": payload.get("tenant_id"),
        "role": payload.get("role"),
    }
    return request.state.user
