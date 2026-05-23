from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.auth import router as auth_router
from app.api.billing import router as billing_router
from app.api.resources import router as resources_router
from app.api.orgs import router as orgs_router
from app.api.projects import router as projects_router
from app.api.onboarding import router as onboarding_router
from bson import ObjectId
from app.core.errors import LicenseError
from app.core.config import settings
from app.db.mongo import get_db
import logging


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.(ngrok-free\.app|ngrok-free\.dev|ngrok\.io|ngrok\.app)|https://.*\.vercel\.app|https://.*\.vercel\.com",
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
    await db.audit_logs.create_index([("tenant_id", 1), ("created_at", -1)])
    await db.projects.create_index([("tenant_id", 1), ("slug", 1)], unique=True)
    await db.projects.create_index([("tenant_id", 1), ("deleted_at", 1)])
    await db.org_invites.create_index("token_hash", unique=True)
    await db.org_invites.create_index([("tenant_id", 1), ("email", 1), ("created_at", -1)])
    await db.org_invites.create_index("expires_at", expireAfterSeconds=0)
    if settings.enable_rate_limiter:
        try:
            redis_client = redis.from_url(settings.redis_url)
            await FastAPILimiter.init(redis_client)
        except RedisConnectionError:
            logger.warning(
                "startup_warning %s",
                {"event": "rate_limiter_disabled", "reason": "redis_unavailable"},
            )



app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(resources_router)
app.include_router(orgs_router)
app.include_router(projects_router)
app.include_router(onboarding_router)



@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}





@app.get("/health/redis")
async def redis_health() -> dict:
    enabled = settings.enable_rate_limiter
    connected = bool(getattr(FastAPILimiter, "redis", None))
    return {
        "enabled": enabled,
        "connected": connected,
        "status": "ok" if (not enabled or connected) else "degraded",
    }
