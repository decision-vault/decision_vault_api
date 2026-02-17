from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.core.config import settings
from app.db.mongo import get_db
from app.services.google_chat_service import _utcnow
from app.middleware.guard import withGuard
from app.services.decision_service import create_decision_if_new
from app.services.google_chat_decision_service import decision_record, should_capture
from app.services.google_chat_service import (
    build_decision_record,
    ensure_tenant_domain_binding,
    get_installation,
    increment_thread_activity,
    license_allows_capture,
    list_spaces_cached,
    parse_event_payload,
    record_webhook_event,
    set_scopes,
    store_installation,
    verify_webhook_signature,
    _utcnow,
)


router = APIRouter(prefix="/api/google-chat", tags=["google-chat-connector"])


@router.post("/install")
async def google_chat_install(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    domain = payload.get("domain")
    if not domain:
        raise HTTPException(status_code=400, detail="Missing domain")
    await ensure_tenant_domain_binding(request.state.tenant_id, domain)
    await store_installation(request.state.tenant_id, domain)
    return {"status": "ok"}


@router.get("/spaces")
async def google_chat_spaces(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    return {"spaces": await list_spaces_cached(request.state.tenant_id)}


@router.post("/scopes")
async def google_chat_scopes(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    spaces = payload.get("spaces", [])
    allow_dm = payload.get("allow_direct_messages", False)
    await set_scopes(request.state.tenant_id, spaces, allow_dm)
    return {"status": "ok"}


@router.get("/status")
async def google_chat_status(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Google Chat not installed")
    return {
        "tenant_id": request.state.tenant_id,
        "domain": installation.get("domain"),
        "installed_at": installation.get("installed_at"),
        "last_webhook_at": installation.get("last_webhook_at"),
        "last_capture_at": installation.get("last_capture_at"),
        "spaces": installation.get("spaces", []),
        "allow_direct_messages": installation.get("allow_direct_messages", False),
    }


@router.get("/threads/activity")
async def google_chat_thread_activity(
    request: Request,
    space: str | None = None,
    limit: int = 50,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    db = get_db()
    query: dict = {"tenant_id": request.state.tenant_id}
    if space:
        query["space"] = space
    cursor = (
        db.google_chat_thread_activity.find(query)
        .sort("last_message_at", -1)
        .limit(limit)
    )
    docs = [doc async for doc in cursor]
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return {"items": docs}


@router.get("/threads/top")
async def google_chat_top_threads(
    request: Request,
    space: str | None = None,
    limit: int = 20,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    db = get_db()
    query: dict = {"tenant_id": request.state.tenant_id}
    if space:
        query["space"] = space
    cursor = (
        db.google_chat_thread_activity.find(query)
        .sort("message_count", -1)
        .limit(limit)
    )
    docs = [doc async for doc in cursor]
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return {"items": docs}


@router.post("/capture")
async def google_chat_capture(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="create_decision", orgRole="admin")),
):
    message = payload.get("message")
    participants = payload.get("participant_count", 0)
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")
    if not should_capture(message, participants):
        raise HTTPException(status_code=400, detail="Message does not match decision rules")
    record = decision_record(message, request.state.tenant_id)
    await create_decision_if_new(record)
    db = get_db()
    await db.google_chat_installations.update_one(
        {"tenant_id": request.state.tenant_id},
        {"$set": {"last_capture_at": _utcnow()}},
    )
    return {"status": "captured"}


@router.post("/webhook")
async def google_chat_webhook(request: Request, background: BackgroundTasks):
    raw_body = await request.body()
    signature = request.headers.get("x-goog-signature", "")
    timestamp = request.headers.get("x-goog-signature-timestamp", "")
    if not verify_webhook_signature(raw_body, signature, timestamp):
        raise HTTPException(status_code=400, detail="Invalid signature")
    token = request.headers.get("x-goog-channel-token")
    if settings.google_chat_webhook_token and token != settings.google_chat_webhook_token:
        raise HTTPException(status_code=400, detail="Invalid token")
    payload = parse_event_payload(raw_body)
    event_id = payload.get("event_id") or payload.get("event", {}).get("eventTime")
    if event_id:
        processed = await record_webhook_event(str(event_id))
        if not processed:
            return {"status": "ok", "idempotent": True}
    background.add_task(process_event, payload)
    return {"status": "ok"}


async def process_event(payload: dict):
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        return
    installation = await get_installation(tenant_id)
    if not installation:
        return
    if not await license_allows_capture(installation["tenant_id"]):
        return
    message = payload.get("message", {})
    space = message.get("space", {}).get("name")
    if space and installation.get("spaces") and space not in installation.get("spaces"):
        return
    if not installation.get("allow_direct_messages", False):
        if message.get("space", {}).get("type") == "DM":
            return
    participant_count = payload.get("participant_count", 0)
    if not should_capture(message, participant_count):
        return
    thread_id = message.get("thread", {}).get("name")
    if thread_id:
        await increment_thread_activity(tenant_id, thread_id, space)
    record = build_decision_record(message, installation["tenant_id"])
    await create_decision_if_new(record)
    db = get_db()
    await db.google_chat_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"last_webhook_at": _utcnow(), "last_capture_at": _utcnow()}},
    )
