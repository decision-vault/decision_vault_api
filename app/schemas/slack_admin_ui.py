from pydantic import BaseModel, Field


class SlackWorkspaceMapping(BaseModel):
    tenant_id: str
    workspace_id: str
    workspace_name: str | None = None
    installed_at: str | None = None


class SlackChannelMapping(BaseModel):
    channel_id: str
    channel_name: str | None = None
    project_id: str | None = None
    project_name: str | None = None


class SlackChannelMappingUpdate(BaseModel):
    channel_id: str
    project_id: str | None = Field(default=None, description="Optional project mapping")


class SlackAdminConfigResponse(BaseModel):
    workspace: SlackWorkspaceMapping
    channel_mappings: list[SlackChannelMapping] = Field(default_factory=list)
