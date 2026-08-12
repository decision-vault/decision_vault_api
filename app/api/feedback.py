from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator

from app.db.mongo import get_db
from app.middleware.guard import withGuard

router = APIRouter(prefix="/api", tags=["feedback"])


def _validate_type(value: str) -> str:
    value = value.strip().lower()
    if value not in ("issue", "idea"):
        raise ValueError("type must be 'issue' or 'idea'")
    return value


def _validate_message(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("message is required")
    if len(value) > 5000:
        raise ValueError("message must be at most 5000 characters")
    return value


class FeedbackIn(BaseModel):
    type: str
    message: str

    _validate_type = field_validator("type")(_validate_type)
    _validate_message = field_validator("message")(_validate_message)


class FeedbackUpdate(BaseModel):
    type: str | None = None
    message: str | None = None

    _validate_type = field_validator("type")(_validate_type)

    @field_validator("message")
    @classmethod
    def _validate_optional_message(cls, value):
        if value is None:
            return value
        return _validate_message(value)


@router.post("/orgs/me/feedback")
async def submit_feedback(
    payload: FeedbackIn,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    db = get_db()
    doc = {
        "tenant_id": ObjectId(request.state.tenant_id),
        "user_id": user.get("user_id"),
        "type": payload.type,
        "message": payload.message,
        "status": "open",
        "created_at": datetime.utcnow(),
    }
    result = await db.feedback.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc["tenant_id"] = str(doc["tenant_id"])
    doc.pop("_id", None)
    return doc


@router.get("/orgs/me/feedback")
async def list_feedback(
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    db = get_db()
    tenant_id = ObjectId(request.state.tenant_id)
    cursor = db.feedback.find({"tenant_id": tenant_id}).sort("created_at", -1)
    items = []
    user_ids = set()
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        doc["tenant_id"] = str(doc["tenant_id"])
        doc.pop("_id", None)
        doc.setdefault("status", "open")
        if doc.get("user_id"):
            user_ids.add(doc["user_id"])
        items.append(doc)

    emails = {}
    if user_ids:
        user_cursor = db.users.find({"_id": {"$in": [ObjectId(uid) for uid in user_ids if ObjectId.is_valid(uid)]}}, {"email": 1})
        async for u in user_cursor:
            emails[str(u["_id"])] = u.get("email")
    for item in items:
        item["user_email"] = emails.get(str(item.get("user_id"))) or None

    return {"feedback": items}


async def _get_feedback_doc(db, tenant_id: str, feedback_id: str) -> dict:
    if not ObjectId.is_valid(feedback_id):
        raise HTTPException(status_code=400, detail="invalid feedback id")
    doc = await db.feedback.find_one(
        {"_id": ObjectId(feedback_id), "tenant_id": ObjectId(tenant_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="feedback not found")
    return doc


@router.patch("/orgs/me/feedback/{feedback_id}")
async def update_feedback(
    feedback_id: str,
    payload: FeedbackUpdate,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    db = get_db()
    doc = await _get_feedback_doc(db, request.state.tenant_id, feedback_id)
    if (doc.get("status") or "open") == "withdrawn":
        raise HTTPException(status_code=400, detail="withdrawn feedback cannot be updated")

    updates: dict = {"updated_at": datetime.utcnow()}
    if payload.type is not None:
        updates["type"] = payload.type
    if payload.message is not None:
        updates["message"] = payload.message

    await db.feedback.update_one({"_id": doc["_id"]}, {"$set": updates})
    updated = await db.feedback.find_one({"_id": doc["_id"]})
    updated["id"] = str(updated["_id"])
    updated["tenant_id"] = str(updated["tenant_id"])
    updated.pop("_id", None)
    return updated


@router.delete("/orgs/me/feedback/{feedback_id}")
async def withdraw_feedback(
    feedback_id: str,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    db = get_db()
    doc = await _get_feedback_doc(db, request.state.tenant_id, feedback_id)
    if (doc.get("status") or "open") == "withdrawn":
        raise HTTPException(status_code=400, detail="feedback already withdrawn")
    await db.feedback.update_one(
        {"_id": doc["_id"]},
        {"$set": {"status": "withdrawn", "withdrawn_at": datetime.utcnow()}},
    )
    return {"id": feedback_id, "status": "withdrawn", "ok": True}
