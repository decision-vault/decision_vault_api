"""Billing orchestration — Stripe integration with a local fallback mode.

When ``settings.stripe_secret_key`` is empty the service records plan changes,
invoices and payment methods directly in MongoDB (via the license service) so
the whole billing flow works without external dependencies. When Stripe keys
are configured it creates Stripe customers, checkout/portal sessions and
processes webhooks instead.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.plans import get_plan, normalize_plan
from app.db.mongo import get_db
from app.services.license_service import (
    change_plan,
    get_billing_overview,
    get_or_create_license,
)


def stripe_enabled() -> bool:
    return bool(settings.stripe_secret_key)


async def _stripe_client():
    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe


async def get_or_create_customer(tenant_id: str, email: str | None) -> str | None:
    if not stripe_enabled():
        return None
    license_doc = await get_or_create_license(tenant_id)
    if license_doc.get("stripe_customer_id"):
        return license_doc["stripe_customer_id"]
    stripe = await _stripe_client()
    customer = stripe.Customer.create(
        email=email or license_doc.get("billing_email") or "",
        metadata={"tenant_id": tenant_id},
    )
    await get_db().licenses.update_one(
        {"tenant_id": __import__("bson").ObjectId(tenant_id)},
        {"$set": {"stripe_customer_id": customer["id"]}},
    )
    return customer["id"]


async def create_checkout_session(
    org_id: str, plan: str, billing_cycle: str, email: str | None
) -> dict:
    """Return a checkout/change URL for a plan.

    Stripe mode returns a real checkout session URL. Local mode returns a
    no-op redirect back to the billing page (the caller then applies the plan
    change directly through the license service).
    """
    plan = normalize_plan(plan)
    plan_doc = get_plan(plan)
    if not plan_doc:
        raise ValueError("Invalid plan")

    if not stripe_enabled():
        return {
            "url": settings.stripe_cancel_url.format(org_id=org_id),
            "local": True,
        }

    customer_id = await get_or_create_customer(org_id, email)
    stripe = await _stripe_client()
    price_key = "price_yearly" if billing_cycle == "yearly" else "price_monthly"
    interval = "year" if billing_cycle == "yearly" else "month"
    # Look up (or create) the Stripe price id for this plan+interval.
    price_id = None
    prices = stripe.Price.list(
        lookup_keys=[f"dv_{plan}_{interval}"], limit=1
    )
    if prices.get("data"):
        price_id = prices["data"][0]["id"]
    else:
        product = stripe.Product.create(
            name=f"DecisionVault {plan_doc['name']}",
            metadata={"plan": plan},
        )
        created = stripe.Price.create(
            product=product["id"],
            unit_amount=int(plan_doc[price_key] * 100),
            currency=settings.stripe_currency,
            recurring={"interval": interval},
            lookup_key=f"dv_{plan}_{interval}",
        )
        price_id = created["id"]

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=settings.stripe_success_url.format(org_id=org_id),
        cancel_url=settings.stripe_cancel_url.format(org_id=org_id),
        metadata={"tenant_id": org_id, "plan": plan},
        client_reference_id=org_id,
    )
    return {"url": session["url"], "local": False}


async def create_portal_session(org_id: str) -> dict:
    if not stripe_enabled():
        return {"url": settings.stripe_cancel_url.format(org_id=org_id), "local": True}
    license_doc = await get_or_create_license(org_id)
    customer_id = license_doc.get("stripe_customer_id")
    if not customer_id:
        raise ValueError("No Stripe customer")
    stripe = await _stripe_client()
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=settings.stripe_cancel_url.format(org_id=org_id),
    )
    return {"url": session["url"], "local": False}


async def handle_webhook(payload: bytes, signature: str | None) -> str | None:
    if not stripe_enabled():
        return None
    stripe = await _stripe_client()
    event = stripe.Webhook.construct_event(
        payload, signature, settings.stripe_webhook_secret
    )
    event_type = event["type"]
    data = event["data"]["object"]
    tenant_id = (data.get("metadata") or {}).get("tenant_id")
    if not tenant_id:
        return event_type

    if event_type == "customer.subscription.updated":
        plan = normalize_plan((data.get("metadata") or {}).get("plan") or "free")
        await change_plan(tenant_id, plan)
    elif event_type == "customer.subscription.deleted":
        await change_plan(tenant_id, "free")
    return event_type
