"""Plan catalog — the single source of truth for plans, features and quotas.

Keep pricing and limits here so the rest of the app (licenses, billing, RBAC
guards, frontend plan selectors) reads from one place.
"""

from __future__ import annotations

from typing import Any

PLAN_IDS = ("free", "lite", "pro")

# Feature flags that can be used to enable/disable product capabilities per plan.
PLAN_FEATURES = {
    "free": [
        "projects",
        "team_invites",
        "docs",
        "tasks",
        "decisions",
        "canvas",
    ],
    "lite": [
        "projects",
        "team_invites",
        "docs",
        "tasks",
        "decisions",
        "canvas",
        "workflows",
        "sprint_planning",
    ],
    "pro": [
        "projects",
        "team_invites",
        "docs",
        "tasks",
        "decisions",
        "canvas",
        "workflows",
        "sprint_planning",
        "advanced_ai",
        "integrations",
        "audit_logs",
        "priority_support",
    ],
}

# Quotas keyed by plan. `None` means unlimited.
PLAN_QUOTAS = {
    "free": {
        "projects": 1,
        "team_members": 3,
        "storage_mb": 100,
        "ai_tokens": 500_000,
    },
    "lite": {
        "projects": 5,
        "team_members": 10,
        "storage_mb": 1024,
        "ai_tokens": 5_000_000,
    },
    "pro": {
        "projects": None,
        "team_members": None,
        "storage_mb": None,
        "ai_tokens": None,
    },
}

PLANS: dict[str, dict[str, Any]] = {
    "free": {
        "id": "free",
        "name": "Free",
        "price_monthly": 0,
        "price_yearly": 0,
        "currency": "USD",
        "interval": "month",
        "description": "For individuals and small teams getting started.",
    },
    "lite": {
        "id": "lite",
        "name": "Lite",
        "price_monthly": 9,
        "price_yearly": 90,
        "currency": "USD",
        "interval": "month",
        "description": "For growing teams that need more projects and members.",
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "price_monthly": 25,
        "price_yearly": 250,
        "currency": "USD",
        "interval": "month",
        "description": "Unlimited projects, advanced AI, integrations and audit logs.",
    },
}

# Plan used when creating the default organization at signup.
DEFAULT_SIGNUP_PLAN = "free"


def get_plan(plan_id: str | None) -> dict[str, Any] | None:
    if not plan_id:
        return None
    return PLANS.get(plan_id.lower())


def is_valid_plan(plan_id: str | None) -> bool:
    return get_plan(plan_id) is not None


def plan_has_feature(plan_id: str | None, feature: str) -> bool:
    plan = get_plan(plan_id)
    if not plan:
        return False
    return feature in PLAN_FEATURES.get(plan["id"], [])


def plan_quota(plan_id: str | None, quota_key: str) -> int | None:
    plan = get_plan(plan_id)
    if not plan:
        return None
    return PLAN_QUOTAS.get(plan["id"], {}).get(quota_key)


def plan_price(plan_id: str | None, yearly: bool = False) -> int:
    plan = get_plan(plan_id)
    if not plan:
        return 0
    return plan["price_yearly"] if yearly else plan["price_monthly"]


def plan_catalog() -> list[dict[str, Any]]:
    return [
        {
            **PLANS[plan_id],
            "features": PLAN_FEATURES[plan_id],
            "quotas": PLAN_QUOTAS[plan_id],
        }
        for plan_id in PLAN_IDS
    ]


def normalize_plan(plan_id: str | None) -> str:
    """Map legacy/unknown plan values (e.g. 'trial') to a known plan id."""
    if is_valid_plan(plan_id):
        return plan_id.lower()
    return DEFAULT_SIGNUP_PLAN
