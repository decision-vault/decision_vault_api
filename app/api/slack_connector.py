from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.middleware.guard import withGuard
from app.schemas.slack_admin_ui import SlackAdminConfigResponse, SlackChannelMappingUpdate
from app.schemas.slack_scoping import SlackChannelScopeResponse, SlackChannelScopeUpdate
from app.services import slack_service
from app.services.slack_admin_service import get_admin_config, set_channel_mapping
from app.services.decision_service import create_decision_if_new


router = APIRouter(prefix="/api/slack", tags=["slack-connector"])


class SlackChannelMessageCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@router.get("/oauth/start")
async def slack_oauth_start(
    tenant_id: str,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    if not settings.slack_client_id:
        raise HTTPException(status_code=500, detail="Slack OAuth not configured")
    url = (
        "https://slack.com/oauth/v2/authorize"
        f"?client_id={settings.slack_client_id}"
        f"&scope=channels:history,channels:read,channels:join,groups:read,chat:write,commands"
        f"&redirect_uri={settings.slack_redirect_uri}"
        f"&state={tenant_id}"
    )
    return RedirectResponse(url=url)


@router.get("/oauth/callback")
async def slack_oauth_callback(code: str, state: str):
    if not settings.slack_client_id or not settings.slack_client_secret:
        raise HTTPException(status_code=500, detail="Slack OAuth not configured")
    if not settings.slack_token_encryption_key:
        raise HTTPException(
            status_code=500,
            detail="Missing DV_SLACK_TOKEN_ENCRYPTION_KEY",
        )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://slack.com/api/oauth.v2.access",
            data={
                "client_id": settings.slack_client_id,
                "client_secret": settings.slack_client_secret,
                "code": code,
                "redirect_uri": settings.slack_redirect_uri,
            },
        )
    data = response.json()
    if not data.get("ok"):
        raise HTTPException(status_code=400, detail="Slack OAuth failed")
    if not data.get("team", {}).get("id"):
        raise HTTPException(status_code=400, detail="Slack OAuth failed: missing team id")
    if not data.get("access_token"):
        raise HTTPException(status_code=400, detail="Slack OAuth failed: missing bot access token")

    tenant_id = state
    try:
        await slack_service.ensure_tenant_workspace_binding(tenant_id, data["team"]["id"])
        await slack_service.store_installation(
            tenant_id=tenant_id,
            team_id=data["team"]["id"],
            team_name=data["team"].get("name"),
            bot_token=data["access_token"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(
        url=f"{settings.frontend_base_url}/organizations?connector=slack&status=connected"
    )


@router.post("/reinstall")
async def slack_reinstall(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    tenant_id = request.state.tenant_id
    url = (
        "https://slack.com/oauth/v2/authorize"
        f"?client_id={settings.slack_client_id}"
        f"&scope=channels:history,channels:read,channels:join,groups:read,chat:write,commands"
        f"&redirect_uri={settings.slack_redirect_uri}"
        f"&state={tenant_id}"
    )
    return {"reinstall_url": url}


@router.post("/channels", response_model=SlackChannelScopeResponse)
async def set_channels(
    request: Request,
    payload: SlackChannelScopeUpdate,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    tenant_id = request.state.tenant_id
    await slack_service.set_channels(tenant_id, payload.channel_ids)
    installation = await slack_service.get_installation_by_tenant(tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Slack not installed")
    return {
        "tenant_id": tenant_id,
        "workspace_id": installation["team_id"],
        "allowed_channel_ids": installation.get("channels", []),
    }


@router.get("/channels", response_model=SlackChannelScopeResponse)
async def list_channels(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    installation = await slack_service.get_installation_by_tenant(request.state.tenant_id)
    if not installation:
        raise HTTPException(status_code=404, detail="Slack not installed")
    if installation.get("revoked_at"):
        raise HTTPException(status_code=400, detail="Slack install revoked")
    channels = await slack_service.list_channels(installation)
    return {
        "tenant_id": request.state.tenant_id,
        "workspace_id": installation["team_id"],
        "allowed_channel_ids": installation.get("channels", []),
        "channels": channels,
    }


@router.get("/channels/{channel_id}/messages")
async def list_channel_messages(
    channel_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    _guard=Depends(withGuard(feature="view_decision", orgRole="member")),
):
    try:
        messages = await slack_service.list_channel_messages(request.state.tenant_id, channel_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"channel_id": channel_id, "messages": messages}


@router.post("/channels/{channel_id}/messages")
async def post_channel_message(
    channel_id: str,
    payload: SlackChannelMessageCreate,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="member")),
):
    try:
        message = await slack_service.post_channel_message(request.state.tenant_id, channel_id, payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"channel_id": channel_id, "message": message}


@router.get("/admin/config", response_model=SlackAdminConfigResponse)
async def slack_admin_config(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    config = await get_admin_config(request.state.tenant_id)
    if not config:
        raise HTTPException(status_code=404, detail="Slack not installed")
    return config


@router.post("/admin/channel-mapping")
async def slack_channel_mapping(
    request: Request,
    payload: SlackChannelMappingUpdate,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    try:
        await set_channel_mapping(request.state.tenant_id, payload.channel_id, payload.project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}


@router.post("/revoke")
async def slack_revoke(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    await slack_service.revoke_installation(request.state.tenant_id, "manual_revoke")
    return {"status": "revoked"}


@router.post("/events")
async def slack_events(request: Request, background: BackgroundTasks):
    raw_body = await request.body()
    timestamp = request.headers.get("x-slack-request-timestamp", "")
    signature = request.headers.get("x-slack-signature", "")
    if not slack_service.verify_request(raw_body, timestamp, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = slack_service.parse_event_payload(raw_body)
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    if payload.get("type") == "app_uninstalled":
        team_id = payload.get("team_id")
        if team_id:
            await slack_service.mark_token_revoked(team_id, "app_uninstalled")
        return {"status": "ok"}

    if payload.get("type") != "event_callback":
        return {"status": "ignored"}

    event = payload.get("event", {})
    background.add_task(process_slack_event, event, payload.get("team_id"))
    return {"status": "ok"}


@router.post("/command")
async def slack_command(request: Request, background: BackgroundTasks):
    raw_body = await request.body()
    timestamp = request.headers.get("x-slack-request-timestamp", "")
    signature = request.headers.get("x-slack-signature", "")
    if not slack_service.verify_request(raw_body, timestamp, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = slack_service.parse_slash_payload(raw_body.decode("utf-8"))
    background.add_task(process_slack_command, payload)
    return {"text": "Decision capture queued.", "response_type": "ephemeral"}


async def process_slack_event(event: dict, team_id: str | None):
    if not team_id:
        return
    installation = await slack_service.get_installation_by_team(team_id)
    if not installation:
        return
    if installation.get("revoked_at"):
        return

    if event.get("channel_type") == "im":
        return

    if not await slack_service.is_channel_allowed(installation, event.get("channel")):
        return

    if not await slack_service.license_allows_capture(installation["tenant_id"]):
        return

    # High-precision capture: require both a decision keyword and a threaded message.
    text = event.get("text") or ""
    if not (slack_service.is_decision_signal(text) and slack_service.is_thread_event(event)):
        return

    record = slack_service.build_decision_record(event, installation["tenant_id"])
    await create_decision_if_new(record)


async def process_slack_command(payload: dict):
    team_id = payload.get("team_id")
    if not team_id:
        return
    if payload.get("command") not in {"/decisionvault", "/decisionvault-capture"}:
        return
    installation = await slack_service.get_installation_by_team(team_id)
    if not installation:
        return
    if installation.get("revoked_at"):
        return

    if not await slack_service.is_channel_allowed(installation, payload.get("channel_id")):
        return

    if not await slack_service.license_allows_capture(installation["tenant_id"]):
        return

    event = {
        "channel": payload.get("channel_id"),
        "thread_ts": payload.get("thread_ts") or payload.get("message_ts"),
        "ts": payload.get("message_ts"),
        "user": payload.get("user_id"),
        "text": payload.get("text"),
    }
    record = slack_service.build_decision_record(event, installation["tenant_id"])
    await create_decision_if_new(record)
