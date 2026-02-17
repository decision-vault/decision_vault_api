from __future__ import annotations

ORG_ROLE_ORDER = {
    "viewer": 1,
    "member": 2,
    "admin": 3,
    "owner": 4,
}

PROJECT_ROLE_ORDER = {
    "viewer": 1,
    "contributor": 2,
    "project_admin": 3,
}

PRODUCT_ROLES = {"superAdmin"}

# Permission matrix for organization-level actions
ORG_PERMISSIONS = {
    "billing.manage": {"owner"},
    "integrations.manage": {"owner", "admin"},
    "decisions.create": {"owner", "admin", "member"},
    "org.read": {"owner", "admin", "member", "viewer"},
}

# Permission matrix for project-level actions
PROJECT_PERMISSIONS = {
    "project.manage": {"project_admin"},
    "project.write": {"project_admin", "contributor"},
    "project.read": {"project_admin", "contributor", "viewer"},
}


def is_super_admin(role: str | None) -> bool:
    return role in PRODUCT_ROLES


def org_role_at_least(role: str | None, required: str) -> bool:
    if role is None:
        return False
    return ORG_ROLE_ORDER.get(role, 0) >= ORG_ROLE_ORDER.get(required, 0)


def project_role_at_least(role: str | None, required: str) -> bool:
    if role is None:
        return False
    return PROJECT_ROLE_ORDER.get(role, 0) >= PROJECT_ROLE_ORDER.get(required, 0)


def org_permission_allows(role: str | None, permission: str) -> bool:
    allowed = ORG_PERMISSIONS.get(permission, set())
    return role in allowed


def project_permission_allows(role: str | None, permission: str) -> bool:
    allowed = PROJECT_PERMISSIONS.get(permission, set())
    return role in allowed
