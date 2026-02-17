from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.middleware.guard import withGuard
from app.services.decision_service import create_decision_if_new
from app.services.zoom_decision_service import should_capture_chat, should_capture_meeting
from app.services.zoom_service import (
    build_decision_record,
    exchange_code_for_tokens,
    ensure_tenant_account_binding,
    get_installation,
    get_installation_by_account,
    license_allows_capture,
    list_chat_channels,
    list_meetings,
    mark_webhook_received,
    parse_webhook_payload,
    record_webhook_event,
    set_scopes,
    store_installation,
    verify_zoom_signature,
)


router = APIRouter(prefix="/api/zoom", tags=["zoom-connector"])


@router.get("/oauth/start")
async def zoom_oauth_start(
    tenant_id: str,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    url = (
        "https://zoom.us/oauth/authorize"
        f"?response_type=code"
        f"&client_id={settings.zoom_client_id}"
        f"&redirect_uri={settings.zoom_redirect_uri}"
        f"&state={tenant_id}"
    )
    return RedirectResponse(url=url)


@router.get("/oauth/callback")
async def zoom_oauth_callback(code: str, state: str):
    tokens = await exchange_code_for_tokens(code)
    account_id = tokens.get("account_id")
    if not account_id:
        raise HTTPException(status_code=400, detail="Missing account_id")
    await ensure_tenant_account_binding(state, account_id)
    await store_installation(state, account_id, tokens)
    return {"status": "ok"}


@router.post("/scopes")
async def zoom_scopes(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    meeting_ids = payload.get("meeting_ids", [])
    chat_channel_ids = payload.get("chat_channel_ids", [])
    allow_chat = payload.get("allow_chat", True)
    await set_scopes(request.state.tenant_id, meeting_ids, chat_channel_ids, allow_chat)
    return {"status": "ok"}


@router.get("/status")
async def zoom_status(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Zoom not installed")
    return {
        "tenant_id": request.state.tenant_id,
        "account_id": installation.get("account_id"),
        "installed_at": installation.get("installed_at"),
        "last_webhook_at": installation.get("last_webhook_at"),
        "webhook_count": installation.get("webhook_count", 0),
        "meeting_ids": installation.get("meeting_ids", []),
        "chat_channel_ids": installation.get("chat_channel_ids", []),
    }


@router.get("/channels")
async def zoom_channels(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Zoom not installed")
    return {"channels": await list_chat_channels(installation)}


@router.get("/meetings")
async def zoom_meetings(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Zoom not installed")
    return {"meetings": await list_meetings(installation)}


@router.post("/capture")
async def manual_capture(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="create_decision", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Zoom not installed")
    source = payload.get("source")
    if source == "meeting":
        meeting_id = payload.get("meeting_id")
        summary = payload.get("summary")
        record = build_decision_record(
            "zoom_meeting",
            request.state.tenant_id,
            {"thread_id": meeting_id, "meeting_id": meeting_id, "text": summary},
        )
        await create_decision_if_new(record)
        return {"status": "captured"}
    if source == "chat":
        channel_id = payload.get("channel_id")
        thread_id = payload.get("thread_id")
        text = payload.get("text")
        record = build_decision_record(
            "zoom_chat",
            request.state.tenant_id,
            {
                "thread_id": thread_id,
                "channel_id": channel_id,
                "message_id": payload.get("message_id"),
                "text": text,
            },
        )
        await create_decision_if_new(record)
        return {"status": "captured"}
    raise HTTPException(status_code=400, detail="Invalid source")


@router.post("/webhook")
async def zoom_webhook(request: Request, background: BackgroundTasks):
    raw_body = await request.body()
    timestamp = request.headers.get("x-zm-request-timestamp", "")
    signature = request.headers.get("x-zm-signature", "")
    if not verify_zoom_signature(raw_body, timestamp, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = parse_webhook_payload(raw_body)
    if payload.get("event") == "endpoint.url_validation":
        plain = payload.get("payload", {}).get("plainToken", "")
        token = plain.encode("utf-8")
        import hmac, hashlib

        encrypted = hmac.new(
            settings.zoom_webhook_secret.encode("utf-8"),
            token,
            hashlib.sha256,
        ).hexdigest()
        return {"plainToken": plain, "encryptedToken": encrypted}

    event_id = payload.get("event_id") or payload.get("event_ts")
    if event_id:
        processed = await record_webhook_event(str(event_id))
        if not processed:
            return {"status": "ok", "idempotent": True}
    background.add_task(process_zoom_event, payload)
    return {"status": "ok"}


async def process_zoom_event(payload: dict):
    event = payload.get("event")
    data = payload.get("payload", {})
    account_id = data.get("account_id")
    if not account_id:
        return
    installation = await get_installation_by_account(account_id)
    if not installation:
        return
    await mark_webhook_received(installation["tenant_id"])
    if not await license_allows_capture(installation["tenant_id"]):
        return

    meeting_ids = installation.get("meeting_ids", [])
    chat_channels = installation.get("chat_channel_ids", [])

    if event == "meeting.summary_completed":
        meeting_id = data.get("object", {}).get("id")
        summary = data.get("object", {}).get("summary")
        if should_capture_meeting(meeting_id, meeting_ids, summary):
            record = build_decision_record(
                "zoom_meeting",
                installation["tenant_id"],
                {"thread_id": meeting_id, "meeting_id": meeting_id, "text": summary},
            )
            await create_decision_if_new(record)

    if event == "im.chat_message_sent":
        message = data.get("object", {})
        channel_id = message.get("channel_id")
        text = message.get("message", "")
        thread_id = message.get("thread_id")
        if should_capture_chat(channel_id, chat_channels, text, thread_id):
            record = build_decision_record(
                "zoom_chat",
                installation["tenant_id"],
                {
                    "thread_id": thread_id,
                    "channel_id": channel_id,
                    "message_id": message.get("id"),
                    "text": text,
                },
            )
            await create_decision_if_new(record)
