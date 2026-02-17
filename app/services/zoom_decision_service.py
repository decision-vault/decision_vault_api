from app.services.zoom_service import is_decision_marker


def should_capture_meeting(meeting_id: str, meeting_ids: list[str], summary: str | None) -> bool:
    if meeting_id not in meeting_ids:
        return False
    if not summary:
        return False
    return is_decision_marker(summary)


def should_capture_chat(channel_id: str, allowed_channels: list[str], text: str, thread_id: str | None) -> bool:
    if channel_id not in allowed_channels:
        return False
    if not thread_id:
        return False
    return is_decision_marker(text)
