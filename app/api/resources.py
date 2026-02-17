from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId

from app.middleware.auth import get_current_user
from app.middleware.rbac import requireOrgRole, requireProjectRole
from app.core.rbac import is_super_admin
from app.schemas.license import License, LicenseCreate, LicenseUpdate
from app.schemas.project_member import (
    ProjectMember,
    ProjectMemberCreate,
    ProjectMemberUpdate,
)
from app.services.license_service import (
    create_license,
    delete_license,
    get_current_license,
    license_status_banner,
    restore_license,
    update_license,
)
from app.services.audit_service import log_event
from app.services.project_member_service import (
    add_project_member,
    list_project_members,
    remove_project_member,
    restore_project_member,
    update_project_member,
)
from app.services.llm_health_service import probe_llm
from app.db.mongo import get_db
from app.utils.serialize import serialize_doc


router = APIRouter(prefix="/api", tags=["resources"])


@router.get("/licenses/current", response_model=License)
async def current_license(user=Depends(requireOrgRole(permission="org.read"))):
    license_doc = await get_current_license(user.get("tenant_id"))
    if not license_doc:
        raise HTTPException(status_code=404, detail="License not found")
    return license_doc


@router.get("/licenses/banner")
async def license_banner(user=Depends(requireOrgRole(permission="org.read"))):
    license_doc = await get_current_license(user.get("tenant_id"))
    if not license_doc:
        raise HTTPException(status_code=404, detail="License not found")
    return license_status_banner(license_doc)


@router.post("/licenses", response_model=License)
async def create_license_route(
    payload: LicenseCreate, user=Depends(requireOrgRole(permission="billing.manage"))
):
    try:
        license_doc = await create_license(
            user.get("tenant_id"), user.get("user_id"), payload.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="license.created",
        entity_type="license",
        entity_id=license_doc.get("_id", ""),
    )
    return license_doc


@router.put("/licenses/{license_id}", response_model=License)
async def update_license_route(
    license_id: str, payload: LicenseUpdate, user=Depends(requireOrgRole(permission="billing.manage"))
):
    updated = await update_license(user.get("tenant_id"), license_id, payload.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="License not found")
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="license.updated",
        entity_type="license",
        entity_id=license_id,
    )
    return updated


@router.delete("/licenses/{license_id}")
async def delete_license_route(
    license_id: str, user=Depends(requireOrgRole(permission="billing.manage"))
):
    deleted = await delete_license(user.get("tenant_id"), license_id, user.get("user_id"))
    if not deleted:
        raise HTTPException(status_code=404, detail="License not found")
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="license.deleted",
        entity_type="license",
        entity_id=license_id,
    )
    return {"status": "deleted"}


@router.post("/licenses/{license_id}/restore", response_model=License)
async def restore_license_route(
    license_id: str, user=Depends(requireOrgRole(permission="billing.manage"))
):
    restored = await restore_license(user.get("tenant_id"), license_id)
    if not restored:
        raise HTTPException(status_code=404, detail="License not found")
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="license.restored",
        entity_type="license",
        entity_id=license_id,
    )
    return restored


@router.get(
    "/projects/{project_id}/members", response_model=list[ProjectMember]
)
async def project_members(
    project_id: str, user=Depends(requireProjectRole(permission="project.read"))
):
    return await list_project_members(user.get("tenant_id"), project_id)


@router.post(
    "/projects/{project_id}/members", response_model=ProjectMember
)
async def add_member(
    project_id: str,
    payload: ProjectMemberCreate,
    user=Depends(requireProjectRole(permission="project.manage")),
):
    member = await add_project_member(
        user.get("tenant_id"), project_id, payload.user_id, payload.role
    )
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="project_member.created",
        entity_type="project_member",
        entity_id=payload.user_id,
        metadata={"project_id": project_id},
    )
    return member


@router.put(
    "/projects/{project_id}/members/{user_id}", response_model=ProjectMember
)
async def update_member(
    project_id: str,
    user_id: str,
    payload: ProjectMemberUpdate,
    user=Depends(requireProjectRole(permission="project.manage")),
):
    updated = await update_project_member(
        user.get("tenant_id"), project_id, user_id, payload.role
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project member not found")
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="project_member.updated",
        entity_type="project_member",
        entity_id=user_id,
        metadata={"project_id": project_id},
    )
    return updated


@router.delete("/projects/{project_id}/members/{user_id}")
async def delete_member(
    project_id: str,
    user_id: str,
    user=Depends(requireProjectRole(permission="project.manage")),
):
    deleted = await remove_project_member(
        user.get("tenant_id"), project_id, user_id, user.get("user_id")
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Project member not found")
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="project_member.deleted",
        entity_type="project_member",
        entity_id=user_id,
        metadata={"project_id": project_id},
    )
    return {"status": "deleted"}


@router.post("/projects/{project_id}/members/{user_id}/restore", response_model=ProjectMember)
async def restore_member(
    project_id: str,
    user_id: str,
    user=Depends(requireProjectRole(permission="project.manage")),
):
    restored = await restore_project_member(user.get("tenant_id"), project_id, user_id)
    if not restored:
        raise HTTPException(status_code=404, detail="Project member not found")
    await log_event(
        tenant_id=user.get("tenant_id"),
        actor_id=user.get("user_id"),
        action="project_member.restored",
        entity_type="project_member",
        entity_id=user_id,
        metadata={"project_id": project_id},
    )
    return restored


@router.get("/audit-logs")
async def audit_logs(
    entity_type: str | None = Query(default=None),
    action: str | None = Query(default=None),
    actor_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, description="ObjectId cursor"),
    user=Depends(requireOrgRole(permission="org.read")),
):
    db = get_db()
    from bson import ObjectId

    query: dict = {"tenant_id": ObjectId(user.get("tenant_id"))}
    if entity_type:
        query["entity_type"] = entity_type
    if action:
        query["action"] = action
    if actor_id:
        if not ObjectId.is_valid(actor_id):
            raise HTTPException(status_code=400, detail="Invalid actor_id")
        query["actor_id"] = ObjectId(actor_id)
    if cursor:
        if not ObjectId.is_valid(cursor):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        query["_id"] = {"$lt": ObjectId(cursor)}

    cursor_q = db.audit_logs.find(query).sort("_id", -1).limit(limit + 1)
    docs = [serialize_doc(doc) async for doc in cursor_q]
    next_cursor = None
    if len(docs) > limit:
        next_cursor = docs[limit]["_id"]
        docs = docs[:limit]
    return {"items": docs, "next_cursor": next_cursor}


@router.get("/admin/llm/health")
async def llm_health(user=Depends(requireOrgRole(permission="integrations.manage"))):
    return await probe_llm()


@router.get("/admin/health/licenses")
async def license_health(
    user=Depends(requireOrgRole(permission="billing.manage")),
):
    db = get_db()
    tenant_id = user.get("tenant_id")
    duplicates = await db.licenses.aggregate(
        [
            {"$match": {"tenant_id": ObjectId(tenant_id), "deleted_at": None}},
            {"$group": {"_id": "$tenant_id", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}},
        ]
    ).to_list(length=1)
    return {
        "tenant_id": tenant_id,
        "duplicate_active_licenses": bool(duplicates),
        "count": duplicates[0]["count"] if duplicates else 1,
        "remediation": "Run python3 backend/scripts/migrate_licenses.py --apply",
    }


@router.get("/admin/health/licenses/duplicates")
async def license_duplicates_global(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, description="ObjectId cursor"),
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")

    db = get_db()
    from bson import ObjectId

    match_stage: dict = {"deleted_at": None}
    if cursor:
        if not ObjectId.is_valid(cursor):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        match_stage["tenant_id"] = {"$lt": ObjectId(cursor)}

    pipeline = [
        {"$match": match_stage},
        {"$group": {"_id": "$tenant_id", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
        {
            "$lookup": {
                "from": "licenses",
                "localField": "_id",
                "foreignField": "tenant_id",
                "as": "tenant_licenses",
            }
        },
        {
            "$project": {
                "tenant_id": "$_id",
                "count": 1,
                "_id": 0,
                "last_seen": {"$max": "$tenant_licenses.created_at"},
            }
        },
        {"$sort": {"count": -1, "tenant_id": 1}},
        {"$limit": limit + 1},
    ]
    duplicates = await db.licenses.aggregate(pipeline).to_list(length=limit + 1)
    next_cursor = None
    if len(duplicates) > limit:
        next_cursor = str(duplicates[limit]["tenant_id"])
        duplicates = duplicates[:limit]
    return {
        "duplicate_tenants": [
            {
                "tenant_id": str(item["tenant_id"]),
                "count": item["count"],
                "last_seen": item.get("last_seen"),
            }
            for item in duplicates
        ],
        "remediation": "Run python3 backend/scripts/migrate_licenses.py --apply",
        "next_cursor": next_cursor,
    }


@router.get("/admin/stripe/last-event")
async def stripe_last_event(
    tenant_id: str | None = Query(default=None),
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")

    db = get_db()
    query = {"action": {"$regex": r"^stripe\\."}}
    if tenant_id:
        from bson import ObjectId

        if not ObjectId.is_valid(tenant_id):
            raise HTTPException(status_code=400, detail="Invalid tenant_id")
        query["tenant_id"] = ObjectId(tenant_id)

    doc = await db.audit_logs.find(query).sort("created_at", -1).limit(1).to_list(1)
    if not doc:
        return {"last_event": None}
    return {"last_event": serialize_doc(doc[0])}


@router.get("/stripe/last-event")
async def stripe_last_event_tenant(
    user=Depends(requireOrgRole(permission="billing.manage")),
):
    db = get_db()
    query = {
        "tenant_id": ObjectId(user.get("tenant_id")),
        "action": {"$regex": r"^stripe\\."},
    }
    doc = await db.audit_logs.find(query).sort("created_at", -1).limit(1).to_list(1)
    if not doc:
        return {"last_event": None}
    return {"last_event": serialize_doc(doc[0])}


@router.get("/stripe/events")
async def stripe_events_tenant(
    limit: int = Query(default=20, ge=1, le=200),
    cursor: str | None = Query(default=None, description="ObjectId cursor"),
    event_type: str | None = Query(default=None, description="e.g. stripe.invoice.payment_failed"),
    from_date: str | None = Query(default=None, description="ISO timestamp"),
    to_date: str | None = Query(default=None, description="ISO timestamp"),
    user=Depends(requireOrgRole(permission="billing.manage")),
):
    db = get_db()
    query: dict = {
        "tenant_id": ObjectId(user.get("tenant_id")),
        "action": {"$regex": r"^stripe\\."},
    }
    if event_type:
        query["action"] = event_type
    if from_date or to_date:
        from datetime import datetime, timezone

        def _parse_ts(value: str, label: str) -> datetime:
            value = value.strip()
            if value.isdigit():
                return datetime.fromtimestamp(int(value), tz=timezone.utc)
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(value)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid {label}")

        time_range: dict = {}
        if from_date:
            time_range["$gte"] = _parse_ts(from_date, "from_date")
        if to_date:
            time_range["$lte"] = _parse_ts(to_date, "to_date")
        if time_range:
            query["created_at"] = time_range
    if cursor:
        if not ObjectId.is_valid(cursor):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        query["_id"] = {"$lt": ObjectId(cursor)}

    cursor_q = db.audit_logs.find(query).sort("_id", -1).limit(limit + 1)
    docs = [serialize_doc(doc) async for doc in cursor_q]
    next_cursor = None
    if len(docs) > limit:
        next_cursor = docs[limit]["_id"]
        docs = docs[:limit]
    return {"items": docs, "next_cursor": next_cursor}


@router.post("/admin/jobs/refresh-slack-channels")
async def refresh_slack_channels(
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")
    db = get_db()
    await db.slack_installations.update_many(
        {},
        {"$set": {"channels_cache": None, "channels_cache_at": None}},
    )
    return {"status": "queued"}


@router.post("/admin/jobs/renew-teams-subscriptions")
async def renew_teams_subscriptions(
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")
    from app.services.teams_subscription_service import renew_due_subscriptions

    renewed = await renew_due_subscriptions()
    return {"status": "ok", "renewed": renewed}


@router.post("/admin/jobs/refresh-zoom-cache")
async def refresh_zoom_cache(
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")
    db = get_db()
    await db.zoom_installations.update_many(
        {},
        {"$set": {"last_webhook_at": None}},
    )
    return {"status": "queued"}


@router.post("/admin/jobs/refresh-zoom-channels")
async def refresh_zoom_channels(
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")
    db = get_db()
    await db.zoom_installations.update_many(
        {},
        {"$set": {"channels_cache": None, "channels_cache_at": None}},
    )
    return {"status": "queued"}


@router.post("/admin/jobs/refresh-google-chat-spaces")
async def refresh_google_chat_spaces(
    user=Depends(get_current_user),
):
    if not is_super_admin(user.get("role")):
        raise HTTPException(status_code=403, detail="Super admin required")
    db = get_db()
    await db.google_chat_installations.update_many(
        {},
        {"$set": {"spaces_cache": None, "spaces_cache_at": None}},
    )
    return {"status": "queued"}
