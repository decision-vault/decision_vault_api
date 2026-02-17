from datetime import datetime, timedelta, timezone

from bson import ObjectId

from app.core.config import settings
from app.db.mongo import get_db
from app.utils.security import hash_password, verify_password
from app.utils.token import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_token,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def _safe_id(value: ObjectId) -> str:
    return str(value)


def _tenant_query(tenant_id: str | None, tenant_slug: str | None) -> dict:
    if tenant_id:
        return {"_id": _oid(tenant_id)}
    if tenant_slug:
        return {"slug": tenant_slug}
    return {}


def _build_trial_license(tenant_id: ObjectId) -> dict:
    start_date = _utcnow()
    return {
        "tenant_id": tenant_id,
        "plan": "trial",
        "status": "active",
        "start_date": start_date,
        "expiry_date": start_date + timedelta(days=settings.trial_days),
        "grace_period_days": settings.trial_grace_days,
        "created_at": start_date,
        "deleted_at": None,
        "deleted_by": None,
    }


def _build_user(
    tenant_id: ObjectId, email: str, password: str | None, role: str, provider: str
) -> dict:
    doc = {
        "tenant_id": tenant_id,
        "email": email.lower(),
        "role": role,
        "provider": provider,
        "created_at": _utcnow(),
        "last_login_at": None,
    }
    if password:
        doc["password_hash"] = hash_password(password)
    return doc


def _build_tenant(name: str, slug: str) -> dict:
    return {
        "name": name,
        "slug": slug,
        "created_at": _utcnow(),
    }


def _refresh_token_doc(
    user_id: ObjectId,
    tenant_id: ObjectId,
    jti: str,
    token_hash: str,
    expires_at: datetime,
    replaced_by: str | None = None,
) -> dict:
    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "jti": jti,
        "token_hash": token_hash,
        "created_at": _utcnow(),
        "expires_at": expires_at,
        "revoked": False,
        "replaced_by": replaced_by,
    }


def _issue_tokens(user: dict) -> dict:
    access_token, expires_in = create_access_token(
        _safe_id(user["_id"]), _safe_id(user["tenant_id"]), user["role"]
    )
    refresh_token, jti, expires_at = create_refresh_token(
        _safe_id(user["_id"]), _safe_id(user["tenant_id"]), user["role"]
    )
    return {
        "access_token": access_token,
        "expires_in": expires_in,
        "refresh_token": refresh_token,
        "refresh_jti": jti,
        "refresh_expires_at": expires_at,
    }


async def signup(tenant_name: str, email: str, password: str) -> dict:
    db = get_db()
    tenant_id = await _create_tenant(db, tenant_name)

    existing = await db.users.find_one({"tenant_id": tenant_id, "email": email.lower()})
    if existing:
        raise ValueError("Email already in use for this tenant")

    user_doc = _build_user(tenant_id, email, password, "owner", "password")
    user = await db.users.insert_one(user_doc)
    user_id = user.inserted_id

    await db.licenses.insert_one(_build_trial_license(tenant_id))

    user_doc["_id"] = user_id
    tokens = _issue_tokens(user_doc)
    await _store_refresh_token(user_doc, tokens)

    return {"user": user_doc, **tokens}


async def login(tenant_id: str | None, tenant_slug: str | None, email: str, password: str) -> dict:
    db = get_db()
    tenant_query = _tenant_query(tenant_id, tenant_slug)
    if not tenant_query:
        raise ValueError("Tenant identifier required")

    tenant = await db.tenants.find_one(tenant_query)
    if not tenant:
        raise ValueError("Tenant not found")

    user = await db.users.find_one({"tenant_id": tenant["_id"], "email": email.lower()})
    if not user:
        raise ValueError("Invalid credentials")

    if "password_hash" not in user or not verify_password(password, user["password_hash"]):
        raise ValueError("Invalid credentials")

    await db.users.update_one(
        {"_id": user["_id"]}, {"$set": {"last_login_at": _utcnow()}}
    )

    tokens = _issue_tokens(user)
    await _store_refresh_token(user, tokens)

    return {"user": user, **tokens}


async def refresh(refresh_token: str) -> dict:
    db = get_db()
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")

    jti = payload.get("jti")
    if not jti:
        raise ValueError("Missing token id")

    token_doc = await db.refresh_tokens.find_one({"jti": jti})
    if not token_doc or token_doc.get("revoked"):
        await _revoke_user_refresh_tokens(payload.get("sub"))
        raise ValueError("Refresh token revoked")

    if token_doc["token_hash"] != hash_token(refresh_token):
        await _revoke_user_refresh_tokens(payload.get("sub"))
        raise ValueError("Refresh token reuse detected")

    user = await db.users.find_one({"_id": _oid(payload["sub"])})
    if not user:
        raise ValueError("User not found")

    if str(user.get("tenant_id")) != payload.get("tenant_id"):
        raise ValueError("Tenant mismatch")

    tokens = _issue_tokens(user)
    await _store_refresh_token(user, tokens)

    await db.refresh_tokens.update_one(
        {"jti": jti},
        {"$set": {"revoked": True, "replaced_by": tokens["refresh_jti"]}},
    )

    return {"user": user, **tokens}


async def logout(refresh_token: str) -> None:
    db = get_db()
    try:
        payload = decode_token(refresh_token)
    except Exception:
        return

    jti = payload.get("jti")
    if not jti:
        return

    await db.refresh_tokens.update_one({"jti": jti}, {"$set": {"revoked": True}})


async def google_login(
    email: str, tenant_id: str | None, tenant_slug: str | None
) -> dict:
    db = get_db()

    tenant_query = _tenant_query(tenant_id, tenant_slug)
    tenant = await db.tenants.find_one(tenant_query) if tenant_query else None
    if not tenant:
        tenant_name = tenant_slug or email.split("@")[0]
        tenant_id = await _create_tenant(db, tenant_name)
        tenant = await db.tenants.find_one({"_id": tenant_id})

    user = await db.users.find_one({"tenant_id": tenant["_id"], "email": email.lower()})
    if not user:
        user_doc = _build_user(tenant["_id"], email, None, "owner", "google")
        user_insert = await db.users.insert_one(user_doc)
        user_doc["_id"] = user_insert.inserted_id
        await db.licenses.insert_one(_build_trial_license(tenant["_id"]))
        user = user_doc

    tokens = _issue_tokens(user)
    await _store_refresh_token(user, tokens)

    return {"user": user, **tokens}


async def _store_refresh_token(user: dict, tokens: dict) -> None:
    db = get_db()
    token_hash = hash_token(tokens["refresh_token"])
    doc = _refresh_token_doc(
        user["_id"],
        user["tenant_id"],
        tokens["refresh_jti"],
        token_hash,
        tokens["refresh_expires_at"],
    )
    await db.refresh_tokens.insert_one(doc)


async def _revoke_user_refresh_tokens(user_id: str | None) -> None:
    if not user_id:
        return
    db = get_db()
    await db.refresh_tokens.update_many(
        {"user_id": _oid(user_id), "revoked": False}, {"$set": {"revoked": True}}
    )


async def _create_tenant(db, name: str) -> ObjectId:
    base_slug = _slugify(name)
    slug = base_slug
    suffix = 1
    while await db.tenants.find_one({"slug": slug}):
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    tenant = await db.tenants.insert_one(_build_tenant(name, slug))
    return tenant.inserted_id
