from __future__ import annotations

from app.db.mongo import get_db
from app.services.decision_service import create_decision_if_new
from app.services.teams_decision_service import should_capture_decision, to_decision_record
from app.services.teams_service import get_access_token
import httpx


async def sync_channel_delta(installation: dict, team_id: str, channel_id: str) -> int:
    db = get_db()
    link_doc = await db.teams_delta_links.find_one(
        {"tenant_id": installation["tenant_id"], "team_id": team_id, "channel_id": channel_id}
    )
    delta_url = link_doc.get("delta_link") if link_doc else None
    if not delta_url:
        delta_url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages/delta"

    token = await get_access_token(installation)
    headers = {"Authorization": f"Bearer {token}"}
    captured = 0

    async with httpx.AsyncClient() as client:
        response = await client.get(delta_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    for message in data.get("value", []):
        if should_capture_decision(message):
            record = to_decision_record(message, installation["tenant_id"])
            created = await create_decision_if_new(record)
            if created:
                captured += 1

    next_link = data.get("@odata.deltaLink")
    if next_link:
        await db.teams_delta_links.update_one(
            {"tenant_id": installation["tenant_id"], "team_id": team_id, "channel_id": channel_id},
            {"$set": {"delta_link": next_link}},
            upsert=True,
        )
    return captured


async def sync_all_scoped_channels(installation: dict) -> int:
    team_ids = installation.get("teams", [])
    channel_ids = installation.get("channels", [])
    total = 0
    for team_id in team_ids:
        for channel_id in channel_ids:
            total += await sync_channel_delta(installation, team_id, channel_id)
    return total
