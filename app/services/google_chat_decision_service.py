from app.services.google_chat_service import is_decision_signal, is_thread_message


def should_capture(message: dict, participant_count: int) -> bool:
    text = message.get("text") or ""
    # High precision: thread message + explicit decision marker + multiple participants.
    return is_thread_message(message) and is_decision_signal(text) and participant_count >= 2


def decision_record(message: dict, tenant_id: str) -> dict:
    from app.services.google_chat_service import build_decision_record

    return build_decision_record(message, tenant_id)
