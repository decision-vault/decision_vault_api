from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.db.mongo import get_db
from app.middleware.auth import get_current_user
from app.schemas.account import AccountOut, AccountUpdate, ChangePasswordRequest
from app.services.audit_service import log_event
from app.utils.security import hash_password, verify_password


router = APIRouter(prefix="/api/account", tags=["account"])


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


async def _build_account(user_id: str) -> dict:
    db = get_db()
    user = await db.users.find_one({"_id": _oid(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tenant = await db.tenants.find_one({"_id": user.get("tenant_id")})

    return {
        "id": str(user["_id"]),
        "email": user.get("email", ""),
        "role": user.get("role", ""),
        "provider": user.get("provider", ""),
        "is_active": user.get("is_active", True) is True,
        "full_name": user.get("full_name"),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
        "tenant_id": str(user.get("tenant_id", "")) if user.get("tenant_id") else "",
        "tenant_name": tenant.get("name", "") if tenant else "",
        "tenant_slug": tenant.get("slug", "") if tenant else "",
    }


@router.get("/me", response_model=AccountOut)
async def get_account(user=Depends(get_current_user)):
    return await _build_account(user["user_id"])


@router.patch("/me", response_model=AccountOut)
async def update_account(
    payload: AccountUpdate,
    user=Depends(get_current_user),
):
    db = get_db()
    updates = {key: value for key, value in payload.model_dump().items() if value is not None}
    if not updates:
        return await _build_account(user["user_id"])

    result = await db.users.update_one(
        {"_id": _oid(user["user_id"])},
        {"$set": updates},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    await log_event(
        tenant_id=user["tenant_id"],
        actor_id=user["user_id"],
        action="account.updated",
        entity_type="user",
        entity_id=user["user_id"],
        metadata={"fields": list(updates.keys())},
    )

    return await _build_account(user["user_id"])


@router.post("/me/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    user=Depends(get_current_user),
):
    db = get_db()
    user_doc = await db.users.find_one({"_id": _oid(user["user_id"])})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    if "password_hash" not in user_doc or not user_doc["password_hash"]:
        raise HTTPException(status_code=400, detail="Password login is not enabled for this account")

    if not verify_password(payload.current_password, user_doc["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    await db.users.update_one(
        {"_id": _oid(user["user_id"])},
        {"$set": {"password_hash": hash_password(payload.new_password)}},
    )

    await log_event(
        tenant_id=user["tenant_id"],
        actor_id=user["user_id"],
        action="account.password_changed",
        entity_type="user",
        entity_id=user["user_id"],
    )

    return {"status": "ok"}
