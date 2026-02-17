import stripe
from bson import ObjectId

from app.core.config import settings
from app.db.mongo import get_db
from app.services.license_service import get_current_license


stripe.api_key = settings.stripe_secret_key


def _price_for_plan(plan: str) -> str:
    if plan == "starter":
        return settings.stripe_price_starter
    if plan == "team":
        return settings.stripe_price_team
    raise ValueError("Unsupported plan")


async def create_checkout_session(
    tenant_id: str, plan: str, success_url: str, cancel_url: str
) -> dict:
    if plan == "enterprise":
        raise ValueError("Enterprise plan is manual")

    license_doc = await get_current_license(tenant_id)
    if not license_doc:
        raise ValueError("License missing")

    price_id = _price_for_plan(plan)
    if not price_id:
        raise ValueError("Stripe price not configured")

    customer_id = license_doc.get("stripe_customer_id")
    if not customer_id:
        customer = stripe.Customer.create(metadata={"tenant_id": tenant_id})
        customer_id = customer.id
        db = get_db()
        await db.licenses.update_one(
            {
                "tenant_id": ObjectId(tenant_id),
                "$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
            },
            {"$set": {"stripe_customer_id": customer_id}},
        )

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        customer=customer_id,
        client_reference_id=tenant_id,
    )
    return {"checkout_url": session.url, "session_id": session.id}
