from bson import ObjectId

from app.db.mongo import get_db


async def get_admin_config(tenant_id: str) -> dict | None:
    db = get_db()
    installation = await db.slack_installations.find_one({"tenant_id": tenant_id})
    if not installation:
        return None

    mappings = await db.slack_channel_mappings.find({"tenant_id": tenant_id}).to_list(length=500)
    project_ids = [ObjectId(m["project_id"]) for m in mappings if m.get("project_id")]
    projects = {}
    if project_ids:
        cursor = db.projects.find({"_id": {"$in": project_ids}, "tenant_id": ObjectId(tenant_id)})
        async for project in cursor:
            projects[str(project["_id"])] = project.get("name")
    mapping_index = {m["channel_id"]: m.get("project_id") for m in mappings}

    return {
        "workspace": {
            "tenant_id": tenant_id,
            "workspace_id": installation["team_id"],
            "workspace_name": installation.get("team_name"),
            "installed_at": installation.get("installed_at"),
        },
        "channel_mappings": [
            {
                "channel_id": channel_id,
                "channel_name": None,
                "project_id": mapping_index.get(channel_id),
                "project_name": projects.get(mapping_index.get(channel_id)) if mapping_index.get(channel_id) else None,
            }
            for channel_id in installation.get("channels", [])
        ],
    }


async def set_channel_mapping(tenant_id: str, channel_id: str, project_id: str | None) -> None:
    db = get_db()
    if project_id:
        project = await db.projects.find_one(
            {"_id": ObjectId(project_id), "tenant_id": ObjectId(tenant_id)}
        )
        if not project:
            raise ValueError("Invalid project_id")
    await db.slack_channel_mappings.update_one(
        {"tenant_id": tenant_id, "channel_id": channel_id},
        {
            "$set": {
                "tenant_id": tenant_id,
                "channel_id": channel_id,
                "project_id": project_id,
            }
        },
        upsert=True,
    )
    await db.slack_installations.update_one(
        {"tenant_id": tenant_id},
        {"$set": {"channels_cache": None, "channels_cache_at": None}},
    )
