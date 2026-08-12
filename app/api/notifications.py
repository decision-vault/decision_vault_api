import logging

import redis.asyncio as redis
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.db.mongo import get_db
from app.middleware.guard import withGuard
from app.services import notification_service as ns
from app.services.tenant_service import user_owns_tenant
from app.utils.token import decode_token

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api", tags=["notifications"])
ws_router = APIRouter(tags=["notifications"])


def _validate_type(value: str) -> str:
    value = (value or "system").strip().lower()
    if value not in ns.NOTIFICATION_TYPES:
        raise ValueError("type must be one of security, performance, messages, system")
    return value


def _validate_title(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("title is required")
    if len(value) > 200:
        raise ValueError("title must be at most 200 characters")
    return value


class NotificationIn(BaseModel):
    type: str = "system"
    title: str
    message: str = ""
    severity: int = Field(default=2, ge=1, le=3)
    project_id: str | None = None

    _validate_type = field_validator("type")(_validate_type)
    _validate_title = field_validator("title")(_validate_title)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value):
        value = (value or "").strip()
        if len(value) > 2000:
            raise ValueError("message must be at most 2000 characters")
        return value


@router.get("/orgs/me/notifications")
async def list_notifications(
    request: Request,
    type: str = "",
    status: str = "",
    severity: int | None = None,
    limit: int = 50,
    user=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    if limit < 1 or limit > 200:
        limit = 50
    items = await ns.list_notifications(
        request.state.tenant_id,
        user["user_id"],
        type_=type or None,
        status=status or None,
        severity=severity,
        limit=limit,
    )
    return {"notifications": items, "unread": await ns.unread_count(
        request.state.tenant_id, user["user_id"]
    )}


@router.get("/orgs/me/notifications/unread-count")
async def get_unread_count(
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    count = await ns.unread_count(
        request.state.tenant_id, user["user_id"]
    )
    return {"unread": count}


@router.post("/orgs/me/notifications")
async def create_notification(
    payload: NotificationIn,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    return await ns.create_notification(
        tenant_id=request.state.tenant_id,
        user_id=user["user_id"],
        type_=payload.type,
        title=payload.title,
        message=payload.message,
        severity=payload.severity,
        project_id=payload.project_id,
    )


@router.patch("/orgs/me/notifications/{notification_id}")
async def mark_read(
    notification_id: str,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    updated = await ns.mark_read(
        request.state.tenant_id, user["user_id"], notification_id
    )
    if not updated:
        raise HTTPException(status_code=404, detail="notification not found")
    return updated


@router.post("/orgs/me/notifications/read-all")
async def mark_all_read(
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    modified = await ns.mark_all_read(
        request.state.tenant_id, user["user_id"]
    )
    return {"ok": True, "updated": modified}


@ws_router.websocket("/ws/notifications")
async def notifications_ws(websocket: WebSocket):
    await websocket.accept()

    token = websocket.query_params.get("token", "")
    tenant_id = websocket.query_params.get("tenant_id", "")
    user_id = None

    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise ValueError("invalid token type")
        user_id = payload.get("sub")
        user = await get_db().users.find_one({"_id": ObjectId(user_id)})
        if not user or user.get("deleted_at") is not None:
            raise ValueError("user not found")
        if tenant_id != payload.get("tenant_id"):
            owned = await user_owns_tenant(user_id, tenant_id)
            if not owned:
                raise ValueError("tenant mismatch")
    except Exception as exc:
        await websocket.send_json({"type": "error", "detail": "unauthorized"})
        await websocket.close(code=4401)
        return

    channel = ns.NOTIFICATION_CHANNEL.format(tenant_id=tenant_id)
    redis_client = redis.from_url(settings.redis_url)
    pubsub = redis_client.pubsub()
    try:
        await pubsub.subscribe(channel)
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            await websocket.send_text(data)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("notification_ws_error")
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await redis_client.aclose()
        except Exception:
            pass
