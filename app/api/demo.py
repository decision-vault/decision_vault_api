from __future__ import annotations

import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, Request
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from app.core.config import settings
from app.db.mongo import get_db
from app.schemas.demo_request import DemoRequestCreate, DemoRequestResponse


router = APIRouter(prefix="/api/demo", tags=["demo"])
logger = logging.getLogger("decisionvault.demo")


async def rate_limit_demo(request: Request):
    if not settings.enable_rate_limiter:
        return
    if not FastAPILimiter.redis:
        return
    limiter = RateLimiter(times=5, seconds=600)
    await limiter(request)


@router.post("/requests", response_model=DemoRequestResponse)
async def create_demo_request(payload: DemoRequestCreate, request: Request, _rl=Depends(rate_limit_demo)):
    # Honeypot: accept but no-op so bots can't easily detect.
    if payload.website and payload.website.strip():
        now = datetime.now(timezone.utc)
        return DemoRequestResponse(status="ok", request_id=str(ObjectId()), created_at=now)

    db = get_db()
    now = datetime.now(timezone.utc)
    doc = {
        "name": payload.name.strip(),
        "email": str(payload.email).strip().lower(),
        "company": payload.company.strip(),
        "role": (payload.role or "").strip() or None,
        "team_size": payload.team_size,
        "notes": (payload.notes or "").strip() or None,
        "preferred_time": (payload.preferred_time or "").strip() or None,
        "timezone": (payload.timezone or "").strip() or None,
        "created_at": now,
        "source": "marketing",
        "ip": getattr(getattr(request, "client", None), "host", None),
        "user_agent": request.headers.get("user-agent"),
        "referer": request.headers.get("referer"),
    }
    res = await db.demo_requests.insert_one(doc)
    logger.info("demo_request_created id=%s company=%s", str(res.inserted_id), doc.get("company"))
    return DemoRequestResponse(request_id=str(res.inserted_id), created_at=now)

