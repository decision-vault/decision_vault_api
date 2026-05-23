from __future__ import annotations

FEATURE_REQUIREMENTS = {
    "upload_document": {"plans": {"trial", "starter", "team", "enterprise"}},
    "manage_integrations": {"plans": {"starter", "team", "enterprise"}},
    "view_decision": {"always": True},
    "search": {"always": True},
}

WRITE_BLOCKED_FEATURES = {
    "upload_document",
    "manage_integrations",
}
