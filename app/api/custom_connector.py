import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.middleware.guard import withGuard
from app.schemas.custom_connector import (
    CustomDecisionPayload,
    CustomDecisionResponse,
    OAuthClientCreate,
    OAuthClientResponse,
    OAuthTokenRequest,
    OAuthTokenResponse,
)
from app.services.custom_connector_service import (
    compute_retry_delay_seconds,
    connector_health,
    create_oauth_access_token,
    create_oauth_client,
    enqueue_retry,
    generate_api_key,
    license_allows_capture,
    record_delivery,
    rotate_oauth_client_secret,
    rotate_api_key,
    verify_oauth_access_token,
    verify_oauth_client,
    verify_api_key,
    verify_hmac_signature,
)
from app.services.decision_service import create_custom_decision
from app.db.mongo import get_db
from app.utils.serialize import serialize_doc
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from bson import ObjectId


router = APIRouter(prefix="/api/custom", tags=["custom-connector"])


async def rate_limit_custom(request: Request):
    if not settings.enable_rate_limiter:
        return
    if not FastAPILimiter.redis:
        return
    limiter = RateLimiter(times=10, seconds=60)
    await limiter(request)


@router.post("/keys")
async def create_key(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    api_key = await generate_api_key(request.state.tenant_id)
    return {"api_key": api_key}


@router.post("/keys/rotate")
async def rotate_key(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    api_key = await rotate_api_key(request.state.tenant_id)
    return {"api_key": api_key}


@router.post("/oauth/clients", response_model=OAuthClientResponse)
async def create_oauth_client_route(
    payload: OAuthClientCreate,
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    client = await create_oauth_client(request.state.tenant_id, payload.name)
    return OAuthClientResponse(**client)


@router.post("/oauth/clients/rotate", response_model=OAuthClientResponse)
async def rotate_oauth_client_route(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    client = await rotate_oauth_client_secret(request.state.tenant_id)
    return OAuthClientResponse(**client)


@router.post("/oauth/token", response_model=OAuthTokenResponse)
async def oauth_token(request: Request):
    if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        payload = OAuthTokenRequest(
            grant_type=form.get("grant_type"),
            client_id=form.get("client_id"),
            client_secret=form.get("client_secret"),
            scope=form.get("scope"),
        )
    else:
        payload = OAuthTokenRequest(**(await request.json()))
    if payload.grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="Unsupported grant type")
    tenant_id = await verify_oauth_client(payload.client_id, payload.client_secret)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    token, expires_in = create_oauth_access_token(tenant_id, payload.client_id)
    return OAuthTokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        scope=payload.scope,
    )


@router.post(
    "/ingest",
    response_model=CustomDecisionResponse,
    dependencies=[Depends(rate_limit_custom)],
)
async def ingest_decision(
    payload: CustomDecisionPayload,
    request: Request,
):
    request_id = str(uuid.uuid4())
    if settings.custom_connector_max_payload_bytes and request.headers.get("content-length"):
        if int(request.headers["content-length"]) > settings.custom_connector_max_payload_bytes:
            await record_delivery(
                payload.tenant_id,
                payload.external_id,
                status="error",
                request_id=request_id,
                error="payload_too_large",
            )
            raise HTTPException(status_code=413, detail="Payload too large")

    api_key = request.headers.get("x-api-key")
    signature = request.headers.get("x-signature")
    bearer = request.headers.get("authorization")
    if bearer and bearer.lower().startswith("bearer "):
        token = bearer.split(" ", 1)[1].strip()
        token_payload = verify_oauth_access_token(token)
        if not token_payload or token_payload.get("tenant_id") != payload.tenant_id:
            await record_delivery(
                payload.tenant_id,
                payload.external_id,
                status="unauthorized",
                request_id=request_id,
                error="invalid_oauth_token",
            )
            raise HTTPException(status_code=401, detail="Invalid OAuth token")
    elif api_key:
        if not await verify_api_key(payload.tenant_id, api_key):
            await record_delivery(
                payload.tenant_id,
                payload.external_id,
                status="unauthorized",
                request_id=request_id,
                error="invalid_api_key",
            )
            raise HTTPException(status_code=401, detail="Invalid API key")
    elif signature:
        raw_body = await request.body()
        if not verify_hmac_signature(raw_body, signature):
            await record_delivery(
                payload.tenant_id,
                payload.external_id,
                status="unauthorized",
                request_id=request_id,
                error="invalid_signature",
            )
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        await record_delivery(
            payload.tenant_id,
            payload.external_id,
            status="unauthorized",
            request_id=request_id,
            error="missing_auth",
        )
        raise HTTPException(status_code=401, detail="Missing authentication")

    db = get_db()
    existing = await db.custom_connector_requests.find_one(
        {"tenant_id": payload.tenant_id, "external_id": payload.external_id}
    )
    if existing:
        await record_delivery(
            payload.tenant_id,
            payload.external_id,
            status="idempotent",
            request_id=request_id,
        )
        return CustomDecisionResponse(status="ok", idempotent=True)
    await db.custom_connector_requests.insert_one(
        {
            "tenant_id": payload.tenant_id,
            "external_id": payload.external_id,
            "created_at": payload.timestamp,
        }
    )

    if not await license_allows_capture(payload.tenant_id):
        await record_delivery(
            payload.tenant_id,
            payload.external_id,
            status="blocked",
            request_id=request_id,
            error="LICENSE_EXPIRED",
        )
        return JSONResponse(
            status_code=403,
            content={"status": "blocked", "error": "LICENSE_EXPIRED"},
        )

    created, decision_id, error_code = await create_custom_decision(payload.model_dump())
    if created:
        await record_delivery(
            payload.tenant_id,
            payload.external_id,
            status="success",
            request_id=request_id,
            metadata={"decision_id": decision_id},
        )
        return CustomDecisionResponse(status="ok", decision_id=decision_id, idempotent=False)
    if error_code == "duplicate":
        await record_delivery(
            payload.tenant_id,
            payload.external_id,
            status="idempotent",
            request_id=request_id,
        )
        return CustomDecisionResponse(status="ok", idempotent=True)
    await record_delivery(
        payload.tenant_id,
        payload.external_id,
        status="error",
        request_id=request_id,
        error="decision_insert_failed",
    )
    await enqueue_retry(payload.model_dump(), error="decision_insert_failed", attempt=1)
    return JSONResponse(
        status_code=202,
        content={"status": "queued", "idempotent": False},
    )


@router.post("/validate")
async def validate_payload(
    payload: CustomDecisionPayload,
    request: Request,
):
    return {"status": "ok", "validated": True}


@router.get("/health")
async def custom_connector_health(
    request: Request,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    return await connector_health(request.state.tenant_id)


@router.get("/deliveries")
async def list_deliveries(
    request: Request,
    status: str | None = None,
    external_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    query: dict = {"tenant_id": request.state.tenant_id}
    if status:
        query["status"] = status
    if external_id:
        query["external_id"] = external_id
    if cursor:
        if not ObjectId.is_valid(cursor):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        query["_id"] = {"$lt": ObjectId(cursor)}
    db = get_db()
    cursor_q = db.custom_connector_deliveries.find(query).sort("_id", -1).limit(limit + 1)
    docs = [serialize_doc(doc) async for doc in cursor_q]
    next_cursor = None
    if len(docs) > limit:
        next_cursor = docs[limit]["_id"]
        docs = docs[:limit]
    return {"items": docs, "next_cursor": next_cursor}


@router.post("/retries/process")
async def process_retries(
    request: Request,
    limit: int = 25,
    _guard=Depends(withGuard(feature="manage_integrations", orgRole="admin")),
):
    db = get_db()
    now = datetime.now(timezone.utc)
    cursor = (
        db.custom_connector_retry_queue.find(
            {"tenant_id": request.state.tenant_id, "next_attempt_at": {"$lte": now}}
        )
        .sort("next_attempt_at", 1)
        .limit(limit)
    )
    retries = [doc async for doc in cursor]
    processed = 0
    for retry in retries:
        payload = retry["payload"]
        created, decision_id, error_code = await create_custom_decision(payload)
        if created or error_code == "duplicate":
            await record_delivery(
                payload["tenant_id"],
                payload["external_id"],
                status="success" if created else "idempotent",
                request_id=f"retry:{retry['_id']}",
                metadata={"decision_id": decision_id},
            )
            await db.custom_connector_retry_queue.delete_one({"_id": retry["_id"]})
            processed += 1
            continue
        attempt = int(retry.get("attempt", 1)) + 1
        if attempt > settings.custom_connector_retry_max_attempts:
            await record_delivery(
                payload["tenant_id"],
                payload["external_id"],
                status="error",
                request_id=f"retry:{retry['_id']}",
                error="retry_exhausted",
            )
            await db.custom_connector_retry_queue.delete_one({"_id": retry["_id"]})
            processed += 1
            continue
        delay = compute_retry_delay_seconds(attempt)
        await db.custom_connector_retry_queue.update_one(
            {"_id": retry["_id"]},
            {
                "$set": {
                    "attempt": attempt,
                    "last_error": error_code or "decision_insert_failed",
                    "next_attempt_at": now + timedelta(seconds=delay),
                    "updated_at": now,
                }
            },
        )
        processed += 1
    return {"status": "ok", "processed": processed}
