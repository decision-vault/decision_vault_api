from fastapi import APIRouter, Depends

from app.middleware.guard import withGuard


router = APIRouter(prefix="/api", tags=["slack"])


@router.get("/orgs/{org_id}/integrations/slack/install")
async def slack_install(
    org_id: str,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    return {"status": "slack install started", "org_id": org_id}


@router.post("/integrations/slack/webhook")
async def slack_webhook(
    _guard=Depends(withGuard(feature="slack_capture", orgRole="admin")),
):
    return {"status": "slack webhook received"}


@router.put("/integrations/{integration_id}/config")
async def slack_config(
    integration_id: str,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    return {"status": "integration config updated", "integration_id": integration_id}
