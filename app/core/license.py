from __future__ import annotations

FEATURE_REQUIREMENTS = {
    "create_decision": {"plans": {"trial", "starter", "team", "enterprise"}},
    "edit_decision": {"plans": {"trial", "starter", "team", "enterprise"}},
    "slack_capture": {"plans": {"trial", "starter", "team", "enterprise"}},
    "upload_document": {"plans": {"trial", "starter", "team", "enterprise"}},
    "manage_integrations": {"plans": {"starter", "team", "enterprise"}},
    "view_decision": {"always": True},
    "search": {"always": True},
}

WRITE_BLOCKED_FEATURES = {
    "create_decision",
    "edit_decision",
    "slack_capture",
    "upload_document",
    "manage_integrations",
}
