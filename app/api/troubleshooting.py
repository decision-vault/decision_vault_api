import re
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, Depends, Request

from app.db.mongo import get_db
from app.middleware.guard import withGuard

router = APIRouter(prefix="/api", tags=["troubleshooting"])

CATEGORIES = [
    "general",
    "account",
    "billing",
    "projects",
    "ai_generation",
    "integrations",
]

DEFAULT_ARTICLES = [
    {
        "category": "ai_generation",
        "title": "AI generation is slow or timing out",
        "summary": "Generation requests can take up to a minute for larger documents. If it times out, retry once before reporting.",
        "steps": [
            "Confirm your connection is stable and retry the request.",
            "Break the generation into smaller pieces instead of one large request.",
            "Check the Usage page to see if you are near the AI token quota.",
            "Wait a few minutes and retry — the generation service may be recovering.",
        ],
        "tags": ["ai", "generation", "slow", "timeout", "retry"],
    },
    {
        "category": "ai_generation",
        "title": "AI token quota reached",
        "summary": "Each plan includes a monthly AI token allowance. When it's used up, AI features are paused until the next cycle.",
        "steps": [
            "Open Usage and confirm which quota you have exceeded.",
            "Wait for the cycle reset, or upgrade to a plan with a higher allowance.",
            "If you upgraded, allow a minute for the new quota to apply.",
        ],
        "tags": ["quota", "limit", "tokens", "ai", "usage"],
    },
    {
        "category": "billing",
        "title": "Plan upgrade is not applying",
        "summary": "In local mode, upgrades are recorded immediately. If the plan still shows the old tier, a refresh usually fixes it.",
        "steps": [
            "Refresh the Billing page.",
            "Confirm the payment method or credit code was accepted.",
            "Check that the status shows 'active' on the Billing page.",
        ],
        "tags": ["billing", "upgrade", "plan", "payment", "credit"],
    },
    {
        "category": "integrations",
        "title": "Integration connection is missing",
        "summary": "GitHub and Vercel integrations are coming soon. Until they launch, no connections can be added.",
        "steps": [
            "Open Integrations and confirm the service you want is listed.",
            "If a service is marked 'Coming soon', it is not available yet.",
            "Contact support if an integration you need is not listed.",
        ],
        "tags": ["github", "vercel", "integration", "connect", "coming soon"],
    },
    {
        "category": "account",
        "title": "A team member cannot access a project",
        "summary": "Access is controlled by the organization team list and the project's permissions.",
        "steps": [
            "Verify the member is invited in the Team page.",
            "Check that they signed in with the invited email.",
            "Confirm the project still exists in the Projects list.",
        ],
        "tags": ["team", "member", "access", "permission", "invite"],
    },
    {
        "category": "projects",
        "title": "Project list is missing a project",
        "summary": "Projects are org-scoped. Make sure you are viewing the correct organization.",
        "steps": [
            "Use the organization switcher to confirm which org you're in.",
            "Refresh the Projects page.",
            "Search by project name in the list.",
        ],
        "tags": ["project", "missing", "list", "org", "organization"],
    },
    {
        "category": "projects",
        "title": "Storage quota is full",
        "summary": "Each plan includes a storage allowance for project files and documents.",
        "steps": [
            "Open Usage to see current storage usage.",
            "Delete or archive unused projects to free space.",
            "Upgrade to a plan with more storage if needed.",
        ],
        "tags": ["storage", "quota", "full", "usage", "space"],
    },
    {
        "category": "general",
        "title": "Feedback is not appearing",
        "summary": "Feedback is stored per organization. Withdrawn items are hidden from the list.",
        "steps": [
            "Confirm you submitted it in the current organization.",
            "Check the Feedback & Issues page and clear any filters.",
            "Withdrawn submissions stay in the database but are not shown.",
        ],
        "tags": ["feedback", "issue", "missing", "withdrawn", "submission"],
    },
]


def _serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = str(doc["_id"])
    doc["tenant_id"] = str(doc["tenant_id"])
    doc.pop("_id", None)
    return doc


async def _ensure_seeded(db, tenant_id: ObjectId) -> None:
    exists = await db.troubleshooting.find_one({"tenant_id": tenant_id})
    if exists:
        return
    now = datetime.utcnow()
    await db.troubleshooting.insert_many(
        [
            {**article, "tenant_id": tenant_id, "created_at": now}
            for article in DEFAULT_ARTICLES
        ]
    )


@router.get("/orgs/me/troubleshooting")
async def list_troubleshooting(
    request: Request,
    q: str = "",
    category: str = "",
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    db = get_db()
    tenant_id = ObjectId(request.state.tenant_id)
    await _ensure_seeded(db, tenant_id)

    query = {"tenant_id": tenant_id}
    if category:
        query["category"] = category

    cursor = db.troubleshooting.find(query).sort("created_at", 1)
    items = []
    async for doc in cursor:
        item = _serialize(doc)
        if q:
            haystack = " ".join(
                [item.get("title", ""), item.get("summary", ""), " ".join(item.get("tags", []))]
            ).lower()
            if re.search(re.escape(q.strip().lower()), haystack) is None:
                continue
        items.append(item)

    return {"articles": items, "categories": CATEGORIES}
