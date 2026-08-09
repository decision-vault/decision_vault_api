from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi_limiter import FastAPILimiter

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.auth import router as auth_router
from app.api.orgs import router as orgs_router
from app.api.projects import router as projects_router
from app.api.tasks import router as tasks_router
from app.api.docs_management import router as docs_management_router
from app.api.prd_generator_routers import router as prd_generator_router
from app.api.workflows import router as workflows_router
from app.api.canvases import router as canvases_router
from app.api.prd_planner import router as prd_planner_router
from app.api.local_workspace import router as local_workspace_router
from app.api.sprint_build import router as sprint_build_router
from app.core.config import settings
from app.db.mongo import get_db

import logging

app = FastAPI(title=settings.app_name)

logger = logging.getLogger("decisionvault.startup")

# =====================================================
# Middleware
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.(ngrok-free\.app|ngrok-free\.dev|ngrok\.io|ngrok\.app)|https://.*\.vercel\.app|https://.*\.vercel\.com",
    allow_credentials=True,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
)

# =====================================================
# Startup
# =====================================================

@app.on_event("startup")
async def startup() -> None:
    db = get_db()

    # -----------------------------
    # Tenant
    # -----------------------------
    await db.tenants.create_index(
        "slug",
        unique=True,
    )

    # -----------------------------
    # Users
    # -----------------------------
    await db.users.create_index(
        [("tenant_id", 1), ("email", 1)],
        unique=True,
    )

    # -----------------------------
    # Refresh Tokens
    # -----------------------------
    await db.refresh_tokens.create_index(
        "jti",
        unique=True,
    )

    await db.refresh_tokens.create_index(
        [("user_id", 1), ("revoked", 1)],
    )

    # -----------------------------
    # Projects
    # -----------------------------
    await db.projects.create_index(
        [("tenant_id", 1), ("slug", 1)],
        unique=True,
    )

    await db.projects.create_index(
        [("tenant_id", 1), ("deleted_at", 1)],
    )

    # -----------------------------
    # Organization Invites
    # -----------------------------
    await db.org_invites.create_index(
        "token_hash",
        unique=True,
    )

    await db.org_invites.create_index(
        [
            ("tenant_id", 1),
            ("email", 1),
            ("created_at", -1),
        ]
    )

    await db.org_invites.create_index(
        "expires_at",
        expireAfterSeconds=0,
    )

    # -----------------------------
    # Audit Logs
    # -----------------------------
    await db.audit_logs.create_index(
        [
            ("tenant_id", 1),
            ("created_at", -1),
        ]
    )

    

    # -----------------------------
    # Rate Limiter
    # -----------------------------
    if settings.enable_rate_limiter:
        try:
            redis_client = redis.from_url(
                settings.redis_url
            )

            await FastAPILimiter.init(
                redis_client
            )

            logger.info(
                "Rate limiter initialized successfully"
            )

        except RedisConnectionError:
            logger.warning(
                "startup_warning %s",
                {
                    "event": "rate_limiter_disabled",
                    "reason": "redis_unavailable",
                },
            )

# =====================================================
# Routers
# =====================================================

app.include_router(auth_router)
app.include_router(orgs_router)
app.include_router(projects_router)
app.include_router(tasks_router)
app.include_router(docs_management_router)
app.include_router(prd_generator_router)
app.include_router(workflows_router)
app.include_router(canvases_router)
app.include_router(prd_planner_router)
app.include_router(local_workspace_router)
app.include_router(sprint_build_router)
# =====================================================
# Health Checks
# =====================================================

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@app.get("/health/redis")
async def redis_health() -> dict:
    enabled = settings.enable_rate_limiter
    connected = bool(
        getattr(
            FastAPILimiter,
            "redis",
            None,
        )
    )

    return {
        "enabled": enabled,
        "connected": connected,
        "status": (
            "ok"
            if (not enabled or connected)
            else "degraded"
        ),
    }
