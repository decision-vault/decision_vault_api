from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.auth import router as auth_router
from app.api.decisions import router as decisions_router
from app.api.billing import router as billing_router
from app.api.example import router as example_router
from app.api.resources import router as resources_router
from app.api.slack import router as slack_router
from app.api.slack_connector import router as slack_connector_router
from app.api.connectors import router as connectors_router
from app.api.uploads import router as uploads_router
from app.api.webhooks import router as webhooks_router
from app.api.why_query import router as why_query_router
from app.api.orgs import router as orgs_router
from app.api.projects import router as projects_router
from app.api.hf_inference import router as hf_inference_router
from app.api.requirements import router as requirements_router
from app.api.messenger import router as messenger_router
from app.api.prd import router as prd_router
from app.services.prd_pg_service import ensure_prd_table
from app.services.llm_usage_service import ensure_usage_table
from app.services.requirements_service import validate_structured, compute_ready_for_prd
from bson import ObjectId
from app.core.errors import LicenseError
from app.core.config import settings
from app.db.mongo import get_db
import logging


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.(ngrok-free\.app|ngrok-free\.dev|ngrok\.io|ngrok\.app)",
    allow_credentials=True,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)
logger = logging.getLogger("decisionvault.startup")


@app.exception_handler(LicenseError)
async def license_error_handler(request: Request, exc: LicenseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )


@app.on_event("startup")
async def startup() -> None:
    db = get_db()
    await db.tenants.create_index("slug", unique=True)
    await db.users.create_index([("tenant_id", 1), ("email", 1)], unique=True)
    await db.refresh_tokens.create_index("jti", unique=True)
    await db.refresh_tokens.create_index([("user_id", 1), ("revoked", 1)])
    active_match = {"$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}]}
    duplicates = await db.licenses.aggregate(
        [
            {"$match": active_match},
            {"$group": {"_id": "$tenant_id", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}},
            {"$project": {"tenant_id": "$_id", "count": 1, "_id": 0}},
            {"$limit": 5},
        ]
    ).to_list(length=5)
    if duplicates:
        dup_ids = [str(item["tenant_id"]) for item in duplicates]
        payload = {
            "event": "license_index_skipped",
            "reason": "duplicate_active_licenses",
            "duplicate_tenant_ids": dup_ids,
            "action": "run_migration",
            "command": "python3 backend/scripts/migrate_licenses.py --apply",
        }
        logger.warning("startup_warning %s", payload)
    else:
        await db.licenses.create_index(
            [("tenant_id", 1), ("deleted_at", 1)],
            unique=True,
        )
    await db.licenses.create_index([("tenant_id", 1), ("status", 1), ("expiry_date", 1)])
    await db.project_members.create_index([("tenant_id", 1), ("project_id", 1), ("user_id", 1)], unique=True)
    await db.audit_logs.create_index([("tenant_id", 1), ("created_at", -1)])
    await db.stripe_events.create_index("event_id", unique=True)
    await db.slack_installations.create_index("tenant_id", unique=True)
    await db.slack_installations.create_index("team_id", unique=True)
    await db.decisions.create_index([("tenant_id", 1), ("thread_ts", 1)], unique=True)
    await db.decisions.create_index([("tenant_id", 1), ("source", 1), ("thread_id", 1)], unique=True)
    await db.decisions.create_index([("tenant_id", 1), ("source", 1), ("external_id", 1)], unique=True)
    await db.slack_channel_mappings.create_index([("tenant_id", 1), ("channel_id", 1)], unique=True)
    await db.projects.create_index([("tenant_id", 1), ("slug", 1)], unique=True)
    await db.projects.create_index([("tenant_id", 1), ("deleted_at", 1)])
    await db.requirements_intakes.create_index([("tenant_id", 1), ("project_id", 1), ("created_at", -1)])
    await db.requirements_intakes.create_index("status")
    # Skip unique index if duplicates exist (first-time boot safety)
    dup_requirements = await db.requirements_intakes.aggregate(
        [
            {"$group": {"_id": {"tenant_id": "$tenant_id", "project_id": "$project_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}},
            {"$limit": 3},
        ]
    ).to_list(length=3)
    if dup_requirements:
        logger.warning(
            "startup_warning %s",
            {
                "event": "requirements_index_skipped",
                "reason": "duplicate_requirements",
                "examples": [
                    {
                        "tenant_id": str(item["_id"]["tenant_id"]),
                        "project_id": str(item["_id"]["project_id"]),
                        "count": item["count"],
                    }
                    for item in dup_requirements
                ],
            },
        )
    else:
        await db.requirements_intakes.create_index([("tenant_id", 1), ("project_id", 1)], unique=True)
    await db.requirements_history.create_index([("tenant_id", 1), ("project_id", 1), ("version", 1)], unique=True)
    await db.prd_documents.create_index([("intake_id", 1), ("version", 1)], unique=True)
    await db.prd_runs.create_index([("tenant_id", 1), ("project_id", 1), ("created_at", -1)])
    await db.prd_runs.create_index([("project_id", 1), ("status", 1), ("updated_at", -1)])
    await db.prd_clarifications.create_index([("tenant_id", 1), ("project_id", 1), ("user_id", 1)], unique=True)
    await db.prd_clarifications.create_index([("project_id", 1), ("updated_at", -1)])
    if settings.enable_rate_limiter:
        try:
            redis_client = redis.from_url(settings.redis_url)
            await FastAPILimiter.init(redis_client)
        except RedisConnectionError:
            logger.warning(
                "startup_warning %s",
                {"event": "rate_limiter_disabled", "reason": "redis_unavailable"},
            )
    await db.project_channels.create_index([("tenant_id", 1), ("project_id", 1), ("slug", 1)], unique=True)
    await db.project_channels.create_index([("tenant_id", 1), ("project_id", 1), ("created_at", 1)])
    await db.project_threads.create_index(
        [("tenant_id", 1), ("project_id", 1), ("channel_id", 1), ("slug", 1)],
        unique=True,
    )
    await db.project_threads.create_index([("tenant_id", 1), ("project_id", 1), ("channel_id", 1), ("created_at", 1)])
    await db.project_messages.create_index(
        [("tenant_id", 1), ("project_id", 1), ("channel_id", 1), ("thread_id", 1), ("created_at", 1)]
    )
    await db.project_channel_favorites.create_index(
        [("tenant_id", 1), ("project_id", 1), ("channel_id", 1), ("user_id", 1)],
        unique=True,
    )
    await db.project_personal_chats.create_index(
        [("tenant_id", 1), ("project_id", 1), ("participant_key", 1)],
        unique=True,
    )
    await db.project_personal_chats.create_index(
        [("tenant_id", 1), ("project_id", 1), ("participant_ids", 1), ("updated_at", -1)]
    )
    await db.project_personal_messages.create_index(
        [("tenant_id", 1), ("project_id", 1), ("chat_id", 1), ("created_at", 1)]
    )
    try:
        await ensure_prd_table()
        await ensure_usage_table()
    except RuntimeError:
        logger.warning(
            "startup_warning %s",
            {"event": "postgres_optional_tables_disabled", "reason": "postgres_unavailable_or_not_configured"},
        )


app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(decisions_router)
app.include_router(example_router)
app.include_router(resources_router)
app.include_router(slack_connector_router)
app.include_router(slack_router)
app.include_router(connectors_router)
app.include_router(uploads_router)
app.include_router(webhooks_router)
app.include_router(why_query_router)
app.include_router(orgs_router)
app.include_router(projects_router)
app.include_router(hf_inference_router)
app.include_router(requirements_router)
app.include_router(messenger_router)
app.include_router(prd_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/debug-readiness/{intake_id}")
async def debug_readiness(intake_id: str) -> dict:
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        return {"missing": [], "low_quality": [], "ready_for_prd": False}
    structured = intake.get("structured") or {}
    missing, low_quality = validate_structured(structured)
    ready = compute_ready_for_prd(structured, missing, low_quality)
    return {"missing": missing, "low_quality": low_quality, "ready_for_prd": ready}


@app.get("/health/redis")
async def redis_health() -> dict:
    enabled = settings.enable_rate_limiter
    connected = bool(getattr(FastAPILimiter, "redis", None))
    return {
        "enabled": enabled,
        "connected": connected,
        "status": "ok" if (not enabled or connected) else "degraded",
    }
