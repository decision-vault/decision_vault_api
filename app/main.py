import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi_limiter import FastAPILimiter

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.auth import router as auth_router
from app.api.account import router as account_router
from app.api.billing import router as billing_router
from app.api.usage import router as usage_router
from app.api.feedback import router as feedback_router
from app.api.troubleshooting import router as troubleshooting_router
from app.api.notifications import router as notifications_router, ws_router as notifications_ws_router
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
from app.api.project_team import router as project_team_router
from app.api.knowledge import router as knowledge_router
from app.core.config import settings
from app.db.mongo import get_db
from app.services.tenant_service import sweep_expired_deleted_tenants

app = FastAPI(title=settings.app_name)

logger = logging.getLogger("decisionvault.startup")


async def _tenant_purge_loop() -> None:
    """Periodically hard-delete organizations past their delete grace period."""
    await asyncio.sleep(settings.tenant_delete_sweep_seconds)
    while True:
        try:
            removed = await sweep_expired_deleted_tenants(settings.tenant_delete_grace_days)
            if removed:
                logger.info("tenant_purge_removed=%s", {"tenant_ids": removed})
        except Exception:
            logger.exception("tenant_purge_sweep_failed")
        await asyncio.sleep(settings.tenant_delete_sweep_seconds)

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

    # Soft-deleted ("paused") organizations: index used by the purge sweep.
    await db.tenants.create_index(
        [("deleted_at", 1)],
    )

    # -----------------------------
    # Tenant purge sweep
    # -----------------------------
    asyncio.create_task(_tenant_purge_loop())

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
    # Project Members
    # -----------------------------
    await db.project_members.create_index(
        [("project_id", 1), ("user_id", 1)],
        unique=True,
        partialFilterExpression={"removed_at": None},
    )

    await db.project_members.create_index(
        [("tenant_id", 1)],
    )

    await db.project_members.create_index(
        [("user_id", 1)],
    )

    # -----------------------------
    # Project Invites
    # -----------------------------
    await db.project_invites.create_index(
        "token_hash",
        unique=True,
    )

    await db.project_invites.create_index(
        [
            ("project_id", 1),
            ("created_at", -1),
        ]
    )

    await db.project_invites.create_index(
        [("email", 1), ("status", 1)],
    )

    await db.project_invites.create_index(
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
    # Usage Daily
    # -----------------------------
    await db.usage_daily.create_index(
        [("tenant_id", 1), ("date", -1), ("project_id", 1)],
    )

    # -----------------------------
    # Feedback / Issues
    # -----------------------------
    await db.feedback.create_index(
        [("tenant_id", 1), ("created_at", -1)],
    )

    # -----------------------------
    # Troubleshooting
    # -----------------------------
    await db.troubleshooting.create_index(
        [("tenant_id", 1), ("category", 1)],
    )

    # -----------------------------
    # Notifications
    # -----------------------------
    await db.notifications.create_index(
        [("tenant_id", 1), ("user_id", 1), ("is_read", 1), ("created_at", -1)],
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
app.include_router(account_router)
app.include_router(billing_router)
app.include_router(usage_router)
app.include_router(feedback_router)
app.include_router(troubleshooting_router)
app.include_router(notifications_router)
app.include_router(notifications_ws_router)
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
app.include_router(project_team_router)
app.include_router(knowledge_router)
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
