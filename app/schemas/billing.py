from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class QuotaUsage(BaseModel):
    limit: Optional[int] = None
    used: int = 0
    remaining: Optional[int] = None


class BillingAddress(BaseModel):
    full_name: Optional[str] = None
    country: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None
    tax_id_type: Optional[str] = None


class PaymentMethodOut(BaseModel):
    id: str
    brand: str = "card"
    last4: str = ""
    exp_month: int = 0
    exp_year: int = 0
    is_default: bool = False


class InvoiceOut(BaseModel):
    id: str
    number: str
    amount: float = 0.0
    currency: str = "USD"
    status: str = "paid"
    created_at: datetime
    download_url: Optional[str] = None


class LicenseOut(BaseModel):
    tenant_id: str
    plan: str
    status: str = "active"
    billing_cycle: str = "monthly"
    start_date: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    grace_period_days: int = 0
    features: list[str] = []
    quotas: dict[str, QuotaUsage] = {}
    usage: dict[str, Any] = {}


class BillingOverview(BaseModel):
    plan: LicenseOut
    plan_name: str
    price: float
    currency: str
    spend_cap_enabled: bool = True
    credit_balance: float = 0.0
    billing_email: Optional[str] = None
    additional_emails: list[str] = []
    address: BillingAddress = BillingAddress()
    invoices: list[InvoiceOut] = []
    payment_methods: list[PaymentMethodOut] = []
    stripe_enabled: bool = False
    stripe_customer_id: Optional[str] = None


class PlanChange(BaseModel):
    plan: str = Field(..., min_length=2, max_length=20)
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|yearly)$")


class BillingUpdate(BaseModel):
    billing_email: Optional[str] = None
    additional_emails: Optional[list[str]] = None
    spend_cap_enabled: Optional[bool] = None
    address: Optional[BillingAddress] = None


class CreditRedeem(BaseModel):
    code: str = Field(..., min_length=1, max_length=64)


class PaymentMethodCreate(BaseModel):
    # Placeholder for local mode; Stripe mode sends a Stripe payment method id.
    token: Optional[str] = None
