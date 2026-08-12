from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import settings
from app.db.mongo import get_db
from app.middleware.guard import withGuard
from app.schemas.billing import (
    BillingAddress,
    BillingOverview,
    BillingUpdate,
    CreditRedeem,
    InvoiceOut,
    LicenseOut,
    PaymentMethodCreate,
    PlanChange,
)
from app.services import billing_service
from app.services.license_service import (
    PlanError,
    QuotaExceededError,
    add_payment_method,
    change_plan,
    get_billing_overview,
    get_or_create_license,
    list_invoices,
    plans_catalog,
    redeem_credit,
    remove_payment_method,
    update_billing_details,
)

router = APIRouter(prefix="/api", tags=["billing"])


@router.get("/orgs/me/billing", response_model=BillingOverview)
async def get_billing(
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    return await get_billing_overview(request.state.tenant_id)


@router.get("/orgs/me/billing/plans")
async def get_plans(
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    return {"plans": plans_catalog()}


@router.get("/orgs/me/billing/invoices", response_model=list[InvoiceOut])
async def get_invoices(
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    return await list_invoices(request.state.tenant_id)


@router.get("/orgs/me/license", response_model=LicenseOut)
async def get_license(
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="viewer")),
):
    overview = await get_billing_overview(request.state.tenant_id)
    return overview["plan"]


@router.patch("/orgs/me/billing", response_model=BillingOverview)
async def patch_billing(
    payload: BillingUpdate,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    return await update_billing_details(request.state.tenant_id, payload.model_dump())


@router.post("/orgs/me/billing/plan", response_model=BillingOverview)
async def post_change_plan(
    payload: PlanChange,
    request: Request,
    user=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    db = get_db()
    owner = await db.users.find_one({"_id": ObjectId(user["user_id"])}, {"email": 1})
    email = (owner or {}).get("email")
    try:
        checkout = await billing_service.create_checkout_session(
            org_id=request.state.tenant_id,
            plan=payload.plan,
            billing_cycle=payload.billing_cycle,
            email=email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if checkout.get("local"):
        try:
            return await change_plan(
                request.state.tenant_id,
                payload.plan,
                payload.billing_cycle,
            )
        except (PlanError, QuotaExceededError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return await get_billing_overview(request.state.tenant_id)


@router.post("/orgs/me/billing/payment-methods", response_model=BillingOverview)
async def add_card(
    payload: PaymentMethodCreate,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    return await add_payment_method(request.state.tenant_id, payload.token)


@router.delete("/orgs/me/billing/payment-methods/{method_id}", response_model=BillingOverview)
async def delete_card(
    method_id: str,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    return await remove_payment_method(request.state.tenant_id, method_id)


@router.post("/orgs/me/billing/credit/redeem", response_model=BillingOverview)
async def redeem(
    payload: CreditRedeem,
    request: Request,
    _guard=Depends(withGuard(feature="edit_decision", orgRole="owner")),
):
    try:
        return await redeem_credit(request.state.tenant_id, payload.code)
    except PlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/billing/webhook")
async def billing_webhook(request: Request):
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=404, detail="Not found")
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        event_type = await billing_service.handle_webhook(payload, signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return {"received": event_type}
