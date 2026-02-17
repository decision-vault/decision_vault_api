from fastapi import APIRouter, HTTPException, Request
import stripe

from app.core.config import settings
from app.services.stripe_webhook_service import (
    handle_checkout_completed,
    handle_invoice_payment_failed,
    handle_invoice_payment_succeeded,
    handle_subscription_deleted,
    handle_subscription_updated,
    record_webhook_event,
)


router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

stripe.api_key = settings.stripe_secret_key


@router.post("/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=settings.stripe_webhook_secret,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    processed = await record_webhook_event(event["id"])
    if not processed:
        return {"status": "ok", "idempotent": True}

    event_type = event["type"]
    if event_type == "checkout.session.completed":
        await handle_checkout_completed(event)
    elif event_type == "invoice.payment_succeeded":
        await handle_invoice_payment_succeeded(event)
    elif event_type == "invoice.payment_failed":
        await handle_invoice_payment_failed(event)
    elif event_type == "customer.subscription.deleted":
        await handle_subscription_deleted(event)
    elif event_type == "customer.subscription.updated":
        await handle_subscription_updated(event)

    return {"status": "ok"}
