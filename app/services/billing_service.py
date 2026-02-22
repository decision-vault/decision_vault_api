import httpx
from bson import ObjectId

from app.core.config import settings
from app.db.mongo import get_db
from app.services.license_service import get_current_license


def _amount_for_plan(plan: str) -> int:
    if plan == "starter":
        return settings.razorpay_amount_starter_paise
    if plan == "team":
        return settings.razorpay_amount_team_paise
    raise ValueError("Unsupported plan")


async def _owner_email(tenant_id: str) -> str | None:
    db = get_db()
    owner = await db.users.find_one(
        {"tenant_id": ObjectId(tenant_id), "role": "owner"},
        sort=[("created_at", 1)],
    )
    return owner.get("email") if owner else None


async def create_checkout_session(
    tenant_id: str, plan: str, success_url: str, cancel_url: str
) -> dict:
    if plan == "enterprise":
        raise ValueError("Enterprise plan is manual")

    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise ValueError("Razorpay not configured")

    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        raise ValueError("License missing")

    amount_paise = _amount_for_plan(plan)
    if amount_paise <= 0:
        raise ValueError(f"Razorpay amount not configured for {plan}")

    payload = {
        "amount": amount_paise,
        "currency": settings.razorpay_currency,
        "description": f"DecisionVault {plan} plan",
        "callback_url": success_url,
        "callback_method": "get",
        "notify": {"email": True, "sms": False},
        "reminder_enable": True,
        "notes": {
            "tenant_id": tenant_id,
            "plan": plan,
            "cancel_url": cancel_url,
        },
    }

    email = await _owner_email(tenant_id)
    if email:
        payload["customer"] = {"email": email}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.razorpay.com/v1/payment_links",
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret),
            json=payload,
        )
        if response.status_code >= 400:
            detail = response.text
            raise ValueError(f"Razorpay checkout failed: {detail}")
        data = response.json()

    checkout_url = data.get("short_url") or data.get("payment_link") or data.get("reference_id")
    if not checkout_url:
        raise ValueError("Razorpay checkout URL missing")

    db = get_db()
    await db.licenses.update_one(
        {
            "tenant_id": ObjectId(tenant_id),
            "$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
        },
        {
            "$set": {
                "billing_provider": "razorpay",
                "razorpay_last_payment_link_id": data.get("id"),
            }
        },
    )

    return {"checkout_url": checkout_url, "session_id": data.get("id", "")}

