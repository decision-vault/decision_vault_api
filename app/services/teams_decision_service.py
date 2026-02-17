from app.services.teams_service import build_decision_record, is_decision_signal, is_threaded_message


def should_capture_decision(message: dict) -> bool:
    text = (message.get("body") or {}).get("content") or ""
    return is_threaded_message(message) and is_decision_signal(text)


def to_decision_record(message: dict, tenant_id: str) -> dict:
    return build_decision_record(message, tenant_id)
