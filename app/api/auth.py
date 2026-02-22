from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from bson import ObjectId
from app.db.mongo import get_db
from app.middleware.auth import get_current_user
from fastapi import Depends

from app.core.config import settings
from app.schemas.auth import LoginRequest, SignupRequest, TokenResponse
from app.services import auth_service
from app.services.audit_service import log_event
from app.utils.token import decode_token


router = APIRouter(prefix="/api/auth", tags=["auth"])


oauth = OAuth()
if settings.google_client_id and settings.google_client_secret:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key="dv_refresh",
        value=refresh_token,
        httponly=True,
        secure=settings.secure_cookies,
        samesite=settings.cookie_samesite,
        max_age=settings.refresh_token_days * 24 * 60 * 60,
        domain=settings.cookie_domain,
        path="/api/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key="dv_refresh",
        domain=settings.cookie_domain,
        path="/api/auth",
    )


@router.post("/signup", response_model=TokenResponse)
async def signup(payload: SignupRequest, response: Response):
    try:
        result = await auth_service.signup(payload.tenant_name, payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _set_refresh_cookie(response, result["refresh_token"])
    await log_event(
        tenant_id=str(result["user"]["tenant_id"]),
        actor_id=str(result["user"]["_id"]),
        action="auth.signup",
        entity_type="user",
        entity_id=str(result["user"]["_id"]),
    )
    return {"access_token": result["access_token"], "expires_in": result["expires_in"]}


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, response: Response):
    try:
        result = await auth_service.login(
            payload.tenant_id, payload.tenant_slug, payload.email, payload.password
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _set_refresh_cookie(response, result["refresh_token"])
    await log_event(
        tenant_id=str(result["user"]["tenant_id"]),
        actor_id=str(result["user"]["_id"]),
        action="auth.login",
        entity_type="user",
        entity_id=str(result["user"]["_id"]),
    )
    return {"access_token": result["access_token"], "expires_in": result["expires_in"]}


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response):
    refresh_token = request.cookies.get("dv_refresh")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")

    try:
        result = await auth_service.refresh(refresh_token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    _set_refresh_cookie(response, result["refresh_token"])
    await log_event(
        tenant_id=str(result["user"]["tenant_id"]),
        actor_id=str(result["user"]["_id"]),
        action="auth.refresh",
        entity_type="user",
        entity_id=str(result["user"]["_id"]),
    )
    return {"access_token": result["access_token"], "expires_in": result["expires_in"]}


@router.post("/logout")
async def logout(request: Request, response: Response):
    refresh_token = request.cookies.get("dv_refresh")
    if refresh_token:
        await auth_service.logout(refresh_token)
        try:
            payload = decode_token(refresh_token)
            await log_event(
                tenant_id=str(payload.get("tenant_id")),
                actor_id=str(payload.get("sub")),
                action="auth.logout",
                entity_type="user",
                entity_id=str(payload.get("sub")),
            )
        except Exception:
            pass
    _clear_refresh_cookie(response)
    return {"status": "ok"}


@router.get("/google")
async def google(request: Request, tenant_id: str | None = None, tenant_slug: str | None = None):
    if "google" not in oauth:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    request.session["tenant_id"] = tenant_id
    request.session["tenant_slug"] = tenant_slug

    redirect_uri = settings.google_redirect_uri
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def google_callback(request: Request):
    if "google" not in oauth:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    user_info = token.get("userinfo")
    if not user_info:
        user_info = await oauth.google.parse_id_token(request, token)

    if not user_info or "email" not in user_info:
        raise HTTPException(status_code=400, detail="Unable to fetch user profile")

    result = await auth_service.google_login(
        user_info["email"],
        request.session.get("tenant_id"),
        request.session.get("tenant_slug"),
    )

    response = RedirectResponse(
        url=f"{settings.frontend_base_url}/auth/callback?access_token={result['access_token']}"
    )
    _set_refresh_cookie(response, result["refresh_token"])
    await log_event(
        tenant_id=str(result["user"]["tenant_id"]),
        actor_id=str(result["user"]["_id"]),
        action="auth.google_login",
        entity_type="user",
        entity_id=str(result["user"]["_id"]),
    )
    return response


@router.get("/session")
async def session(user=Depends(get_current_user)):
    db = get_db()
    user_doc = await db.users.find_one({"_id": ObjectId(user["user_id"])})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    tenant_doc = await db.tenants.find_one({"_id": ObjectId(user["tenant_id"])})
    return {
        "user_id": str(user_doc["_id"]),
        "tenant_id": str(user_doc["tenant_id"]),
        "email": user_doc.get("email", ""),
        "role": user_doc.get("role", ""),
        "provider": user_doc.get("provider", ""),
        "last_login_at": user_doc.get("last_login_at"),
        "tenant_name": tenant_doc.get("name", "") if tenant_doc else "",
        "tenant_slug": tenant_doc.get("slug", "") if tenant_doc else "",
    }
