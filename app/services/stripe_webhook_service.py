from datetime import datetime, timezone

import stripe
from bson import ObjectId

from app.core.config import settings
from app.db.mongo import get_db
from app.services.audit_service import log_event


stripe.api_key = settings.stripe_secret_key


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str | None) -> ObjectId | None:
    if not value or not ObjectId.is_valid(value):
        return None
    return ObjectId(value)


def _plan_from_price(price_id: str | None) -> str | None:
    if not price_id:
        return None
    if price_id == settings.stripe_price_starter:
        return "starter"
    if price_id == settings.stripe_price_team:
        return "team"
    return None


def _subscription_period_end(subscription: dict | None) -> datetime | None:
    if not subscription:
        return None
    period_end_ts = subscription.get("current_period_end")
    if not period_end_ts:
        return None
    return datetime.fromtimestamp(period_end_ts, tz=timezone.utc)


async def record_webhook_event(event_id: str) -> bool:
    db = get_db()
    existing = await db.stripe_events.find_one({"event_id": event_id})
    if existing:
        return False
    await db.stripe_events.insert_one({"event_id": event_id, "created_at": _utcnow()})
    return True


async def _license_for_customer(customer_id: str) -> dict | None:
    db = get_db()
    return await db.licenses.find_one({"stripe_customer_id": customer_id, "deleted_at": None})


async def handle_checkout_completed(event: dict) -> None:
    data = event.get("data", {}).get("object", {})
    customer_id = data.get("customer")
    subscription_id = data.get("subscription")
    latest_invoice_id = data.get("invoice")
    if not customer_id or not subscription_id:
        return

    # Webhooks are the source of truth. We only resolve the tenant via stripe_customer_id.
    subscription = stripe.Subscription.retrieve(subscription_id)
    price_id = None
    if subscription and subscription.get("items"):
        items = subscription["items"]["data"]
        if items:
            price_id = items[0]["price"]["id"]
    plan = _plan_from_price(price_id)
    period_end = _subscription_period_end(subscription) or _utcnow()

    db = get_db()
    license_doc = await _license_for_customer(customer_id)
    if not license_doc:
        return

    # Successful checkout transitions license to active.
    await db.licenses.update_one(
        {"_id": license_doc["_id"]},
        {
            "$set": {
                "plan": plan or license_doc.get("plan"),
                "status": "active",
                "expiry_date": period_end,
                "grace_start_date": None,
                "stripe_subscription_id": subscription_id,
                "stripe_subscription_status": subscription.get("status"),
                "stripe_latest_invoice_id": latest_invoice_id,
            }
        },
    )

    await log_event(
        tenant_id=str(license_doc["tenant_id"]),
        actor_id=str(license_doc["tenant_id"]),
        action="stripe.checkout.completed",
        entity_type="license",
        entity_id=str(license_doc["_id"]),
        metadata={
            "stripe_event_id": event.get("id"),
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "stripe_latest_invoice_id": latest_invoice_id,
            "stripe_subscription_status": subscription.get("status"),
            "stripe_price_id": price_id,
        },
    )


async def handle_invoice_payment_failed(event: dict) -> None:
    data = event.get("data", {}).get("object", {})
    customer_id = data.get("customer")
    latest_invoice_id = data.get("id")
    subscription_id = data.get("subscription")
    subscription_status = data.get("subscription_status")
    if not customer_id:
        return

    license_doc = await _license_for_customer(customer_id)
    if not license_doc:
        return

    # Active -> grace on payment failure. (Read-only enforced elsewhere.)
    if license_doc.get("status") == "active":
        db = get_db()
        await db.licenses.update_one(
            {"_id": license_doc["_id"]},
            {"$set": {"status": "grace", "grace_start_date": _utcnow()}},
        )

    await log_event(
        tenant_id=str(license_doc["tenant_id"]),
        actor_id=str(license_doc["tenant_id"]),
        action="stripe.invoice.payment_failed",
        entity_type="license",
        entity_id=str(license_doc["_id"]),
        metadata={
            "stripe_event_id": event.get("id"),
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "stripe_latest_invoice_id": latest_invoice_id,
            "stripe_subscription_status": subscription_status,
        },
    )


async def handle_invoice_payment_succeeded(event: dict) -> None:
    data = event.get("data", {}).get("object", {})
    customer_id = data.get("customer")
    subscription_id = data.get("subscription")
    latest_invoice_id = data.get("id")
    if not customer_id or not subscription_id:
        return

    license_doc = await _license_for_customer(customer_id)
    if not license_doc:
        return

    subscription = stripe.Subscription.retrieve(subscription_id)
    period_end = _subscription_period_end(subscription) or _utcnow()

    # Grace -> active on successful payment.
    if license_doc.get("status") == "grace":
        db = get_db()
        await db.licenses.update_one(
            {"_id": license_doc["_id"]},
            {
                "$set": {
                    "status": "active",
                    "expiry_date": period_end,
                    "grace_start_date": None,
                    "stripe_subscription_id": subscription_id,
                    "stripe_subscription_status": subscription.get("status"),
                    "stripe_latest_invoice_id": latest_invoice_id,
                }
            },
        )

    await log_event(
        tenant_id=str(license_doc["tenant_id"]),
        actor_id=str(license_doc["tenant_id"]),
        action="stripe.invoice.payment_succeeded",
        entity_type="license",
        entity_id=str(license_doc["_id"]),
        metadata={
            "stripe_event_id": event.get("id"),
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "stripe_latest_invoice_id": latest_invoice_id,
            "stripe_subscription_status": subscription.get("status"),
        },
    )


async def handle_subscription_deleted(event: dict) -> None:
    data = event.get("data", {}).get("object", {})
    subscription_id = data.get("id")
    customer_id = data.get("customer")
    subscription_status = data.get("status")
    if not subscription_id or not customer_id:
        return

    license_doc = await _license_for_customer(customer_id)
    if not license_doc:
        return

    db = get_db()
    # Subscription canceled -> expired.
    await db.licenses.update_one(
        {"_id": license_doc["_id"]},
        {"$set": {"status": "expired", "expiry_date": _utcnow()}},
    )

    await log_event(
        tenant_id=str(license_doc["tenant_id"]),
        actor_id=str(license_doc["tenant_id"]),
        action="stripe.subscription.deleted",
        entity_type="license",
        entity_id=str(license_doc["_id"]),
        metadata={
            "stripe_event_id": event.get("id"),
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "stripe_subscription_status": subscription_status,
        },
    )


async def handle_subscription_updated(event: dict) -> None:
    data = event.get("data", {}).get("object", {})
    subscription_id = data.get("id")
    customer_id = data.get("customer")
    if not subscription_id or not customer_id:
        return

    license_doc = await _license_for_customer(customer_id)
    if not license_doc:
        return

    price_id = None
    if data.get("items"):
        items = data["items"]["data"]
        if items:
            price_id = items[0]["price"]["id"]
    plan = _plan_from_price(price_id)

    db = get_db()
    # Subscription updated -> plan update only (no client-provided plan).
    await db.licenses.update_one(
        {"_id": license_doc["_id"]},
        {
            "$set": {
                "plan": plan or license_doc.get("plan"),
                "stripe_subscription_id": subscription_id,
                "stripe_subscription_status": data.get("status"),
            }
        },
    )

    await log_event(
        tenant_id=str(license_doc["tenant_id"]),
        actor_id=str(license_doc["tenant_id"]),
        action="stripe.subscription.updated",
        entity_type="license",
        entity_id=str(license_doc["_id"]),
        metadata={
            "stripe_event_id": event.get("id"),
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "stripe_subscription_status": data.get("status"),
            "stripe_price_id": price_id,
        },
    )
