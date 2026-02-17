import types
from datetime import datetime, timedelta, timezone

import pytest

from app.services import stripe_webhook_service as svc


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    async def find_one(self, query):
        for doc in self.docs:
            ok = True
            for key, value in query.items():
                if key not in doc:
                    ok = False
                    break
                if doc[key] != value:
                    ok = False
                    break
            if ok:
                return doc
        return None

    async def update_one(self, query, update):
        doc = await self.find_one(query)
        if not doc:
            return
        for key, value in update.get("$set", {}).items():
            doc[key] = value

    async def insert_one(self, doc):
        self.docs.append(doc)

    def find(self, query):
        return []


class FakeDB:
    def __init__(self, licenses):
        self.licenses = FakeCollection(licenses)
        self.stripe_events = FakeCollection([])
        self.audit_logs = FakeCollection([])


@pytest.mark.asyncio
async def test_payment_failed_moves_active_to_grace(monkeypatch):
    license_doc = {
        "_id": "lic1",
        "tenant_id": "t1",
        "stripe_customer_id": "cus_123",
        "status": "active",
        "deleted_at": None,
    }
    db = FakeDB([license_doc])

    monkeypatch.setattr(svc, "get_db", lambda: db)
    monkeypatch.setattr(svc, "log_event", lambda **kwargs: None)

    event = {
        "id": "evt_1",
        "data": {
            "object": {
                "customer": "cus_123",
                "id": "in_1",
                "subscription": "sub_1",
                "subscription_status": "past_due",
            }
        },
    }

    await svc.handle_invoice_payment_failed(event)
    assert license_doc["status"] == "grace"
    assert "grace_start_date" in license_doc


@pytest.mark.asyncio
async def test_payment_succeeded_restores_grace(monkeypatch):
    license_doc = {
        "_id": "lic1",
        "tenant_id": "t1",
        "stripe_customer_id": "cus_123",
        "status": "grace",
        "deleted_at": None,
    }
    db = FakeDB([license_doc])

    monkeypatch.setattr(svc, "get_db", lambda: db)
    monkeypatch.setattr(svc, "log_event", lambda **kwargs: None)

    async def fake_retrieve(sub_id):
        return {
            "current_period_end": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
            "status": "active",
        }

    monkeypatch.setattr(svc.stripe, "Subscription", types.SimpleNamespace(retrieve=fake_retrieve))

    event = {
        "id": "evt_2",
        "data": {"object": {"customer": "cus_123", "subscription": "sub_1", "id": "in_2"}},
    }

    await svc.handle_invoice_payment_succeeded(event)
    assert license_doc["status"] == "active"
    assert "expiry_date" in license_doc


@pytest.mark.asyncio
async def test_checkout_completed_sets_plan(monkeypatch):
    license_doc = {
        "_id": "lic1",
        "tenant_id": "t1",
        "stripe_customer_id": "cus_123",
        "plan": "trial",
        "status": "trial",
        "deleted_at": None,
    }
    db = FakeDB([license_doc])

    monkeypatch.setattr(svc, "get_db", lambda: db)
    monkeypatch.setattr(svc, "log_event", lambda **kwargs: None)

    async def fake_retrieve(sub_id):
        return {
            "items": {"data": [{"price": {"id": svc.settings.stripe_price_starter or "price_starter"}}]},
            "current_period_end": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
            "status": "active",
        }

    monkeypatch.setattr(svc.stripe, "Subscription", types.SimpleNamespace(retrieve=fake_retrieve))

    event = {
        "id": "evt_3",
        "data": {
            "object": {
                "customer": "cus_123",
                "subscription": "sub_1",
                "invoice": "in_3",
            }
        },
    }

    await svc.handle_checkout_completed(event)
    assert license_doc["status"] == "active"
