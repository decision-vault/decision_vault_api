from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import settings
from app.db.mongo import get_db
from app.middleware.guard import withGuard
from app.services import slack_service

router = APIRouter(prefix="/api/connectors", tags=["connectors"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


@router.get("/status")
async def connectors_status(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    db = get_db()
    tenant_id = request.state.tenant_id
    tenant_oid = _oid(tenant_id)

    slack = await db.slack_installations.find_one({"tenant_id": {"$in": [tenant_oid, tenant_id]}})
    return {
        "items": [
            {
                "provider": "slack",
                "connected": bool(slack and not slack.get("revoked_at")),
                "connected_at": slack.get("installed_at") if slack else None,
                "status": "connected" if (slack and not slack.get("revoked_at")) else "not_connected",
            },
        ]
    }


@router.get("/start-url/{provider}")
async def connector_start_url(
    provider: str,
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    tenant_id = request.state.tenant_id
    if provider == "slack":
        if not settings.slack_client_id:
            raise HTTPException(status_code=400, detail="Slack connector not configured: missing DV_SLACK_CLIENT_ID")
        if not settings.slack_redirect_uri:
            raise HTTPException(status_code=400, detail="Slack connector not configured: missing DV_SLACK_REDIRECT_URI")
        return {
            "provider": provider,
            "start_url": (
                "https://slack.com/oauth/v2/authorize"
                f"?client_id={settings.slack_client_id}"
                f"&scope=channels:history,channels:read,channels:join,groups:read,chat:write,commands"
                f"&redirect_uri={settings.slack_redirect_uri}"
                f"&state={tenant_id}"
            ),
        }
    raise HTTPException(status_code=404, detail="Only slack connector is enabled")


@router.post("/disconnect/{provider}")
async def connector_disconnect(
    provider: str,
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    tenant_id = request.state.tenant_id

    if provider == "slack":
        await slack_service.revoke_installation(tenant_id, "manual_disconnect")
        return {"status": "disconnected", "provider": provider}
    raise HTTPException(status_code=404, detail="Only slack connector is enabled")
