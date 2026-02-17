from pydantic import BaseModel, Field


class SlackChannelScopeUpdate(BaseModel):
    channel_ids: list[str] = Field(..., description="Allowed Slack channel IDs")


class SlackChannelScopeResponse(BaseModel):
    tenant_id: str
    workspace_id: str
    allowed_channel_ids: list[str]
    channels: list[dict] | None = None
