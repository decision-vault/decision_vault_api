from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.db.mongo import get_db
from app.middleware.guard import withGuard
from app.services.decision_service import create_decision_if_new
from app.services.teams_decision_service import should_capture_decision, to_decision_record
from app.services.teams_delta_service import sync_all_scoped_channels, sync_channel_delta
from app.services.teams_service import (
    exchange_code_for_tokens,
    get_installation,
    get_installation_by_aad,
    ensure_tenant_org_binding,
    get_message,
    list_channels,
    list_teams,
    license_allows_capture,
    mark_delta_sync,
    mark_webhook_received,
    parse_webhook_payload,
    set_scopes,
    store_installation,
)
from app.services.teams_subscription_service import create_subscription, store_subscription


router = APIRouter(prefix="/api/teams", tags=["teams-connector"])


@router.get("/oauth/start")
async def teams_oauth_start(
    tenant_id: str,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    url = (
        f"https://login.microsoftonline.com/{settings.teams_tenant_id}/oauth2/v2.0/authorize"
        f"?client_id={settings.teams_client_id}"
        f"&response_type=code"
        f"&redirect_uri={settings.teams_redirect_uri}"
        f"&response_mode=query"
        f"&scope=https%3A%2F%2Fgraph.microsoft.com%2F.default"
        f"&state={tenant_id}"
        f"&prompt=admin_consent"
    )
    return RedirectResponse(url=url)


@router.get("/oauth/callback")
async def teams_oauth_callback(code: str, state: str, tenant: str):
    tokens = await exchange_code_for_tokens(code)
    await ensure_tenant_org_binding(state, tenant)
    await store_installation(state, tenant, tokens)
    installation = await get_installation(state)
    if installation:
        subscription = await create_subscription(
            installation,
            settings.teams_redirect_uri.replace("/oauth/callback", "/webhook"),
        )
        await store_subscription(state, subscription)
    return {"status": "ok"}


@router.post("/scopes")
async def update_scopes(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    team_ids = payload.get("team_ids", [])
    channel_ids = payload.get("channel_ids", [])
    allow_private = payload.get("allow_private", False)
    await set_scopes(request.state.tenant_id, team_ids, channel_ids, allow_private)
    return {"status": "ok"}


@router.get("/teams")
async def teams_list(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Teams not installed")
    return {"teams": await list_teams(installation)}


@router.get("/channels")
async def teams_channels(
    request: Request,
    team_id: str,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Teams not installed")
    return {"channels": await list_channels(installation, team_id)}


@router.post("/capture")
async def manual_capture(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="create_decision", orgRole="admin")),
):
    team_id = payload.get("team_id")
    channel_id = payload.get("channel_id")
    message_id = payload.get("message_id")
    if not team_id or not channel_id or not message_id:
        raise HTTPException(status_code=400, detail="Missing identifiers")
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Teams not installed")
    message = await get_message(installation, team_id, channel_id, message_id)
    if not should_capture_decision(message):
        raise HTTPException(status_code=400, detail="Message does not match decision rules")
    record = to_decision_record(message, request.state.tenant_id)
    await create_decision_if_new(record)
    return {"status": "captured"}


@router.post("/sync/delta")
async def delta_sync(
    request: Request,
    payload: dict,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Teams not installed")
    team_id = payload.get("team_id")
    channel_id = payload.get("channel_id")
    if team_id and channel_id:
        captured = await sync_channel_delta(installation, team_id, channel_id)
        await mark_delta_sync(request.state.tenant_id, captured)
        return {"status": "ok", "captured": captured}
    captured = await sync_all_scoped_channels(installation)
    await mark_delta_sync(request.state.tenant_id, captured)
    return {"status": "ok", "captured": captured}


@router.get("/status")
async def teams_status(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    db = get_db()
    installation = await get_installation(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Teams not installed")
    subscription = await db.teams_subscriptions.find_one({"tenant_id": request.state.tenant_id})
    return {
        "tenant_id": request.state.tenant_id,
        "aad_tenant_id": installation.get("aad_tenant_id"),
        "installed_at": installation.get("installed_at"),
        "last_webhook_at": installation.get("last_webhook_at"),
        "last_webhook_count": installation.get("last_webhook_count"),
        "last_delta_sync_at": installation.get("last_delta_sync_at"),
        "last_delta_sync_captured": installation.get("last_delta_sync_captured"),
        "subscription": subscription,
    }


@router.post("/webhook")
async def teams_webhook(request: Request, background: BackgroundTasks):
    raw_body = await request.body()
    if "validationToken" in request.query_params:
        return request.query_params["validationToken"]

    payload = parse_webhook_payload(raw_body)
    if payload.get("value"):
        for notification in payload["value"]:
            if notification.get("clientState") != "decisionvault":
                raise HTTPException(status_code=400, detail="Invalid clientState")
    background.add_task(process_notification, payload)
    return {"status": "ok"}


async def process_notification(payload: dict):
    notifications = payload.get("value", [])
    if notifications:
        aad_tenant_id = notifications[0].get("tenantId")
        if aad_tenant_id:
            installation = await get_installation_by_aad(aad_tenant_id)
            if installation:
                await mark_webhook_received(installation["tenant_id"], len(notifications))
    for notification in notifications:
        aad_tenant_id = notification.get("tenantId")
        if not aad_tenant_id:
            continue
        installation = await get_installation_by_aad(aad_tenant_id)
        if not installation:
            continue
        team_ids = installation.get("teams", [])
        channel_ids = installation.get("channels", [])
        allow_private = installation.get("allow_private", False)
        if not allow_private and notification.get("resource", "").startswith("/chats/"):
            continue
        if not await license_allows_capture(installation["tenant_id"]):
            continue
        message = notification.get("resourceData", {})
        if team_ids and message.get("teamId") not in team_ids:
            continue
        if channel_ids and message.get("channelIdentity", {}).get("channelId") not in channel_ids:
            continue
        if not should_capture_decision(message):
            continue
        record = to_decision_record(message, installation["tenant_id"])
        await create_decision_if_new(record)
