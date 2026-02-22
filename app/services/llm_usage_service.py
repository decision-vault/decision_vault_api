from __future__ import annotations

from datetime import datetime, timezone
from app.db.mongo import get_db


MODEL_COST_PER_1K = {
    "gpt-4o-mini": 0.0006,
    "gpt-4.1": 0.01,
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    per_1k = MODEL_COST_PER_1K.get(model, 0.002)
    return round(((input_tokens + output_tokens) / 1000.0) * per_1k, 6)


async def ensure_usage_table() -> None:
    db = get_db()
    await db.llm_usage_logs.create_index([("tenant_id", 1), ("created_at", -1)])
    await db.llm_usage_logs.create_index([("feature", 1), ("created_at", -1)])


async def log_llm_usage(
    *,
    feature: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    tenant_id: str,
) -> None:
    db = get_db()
    cost = estimate_cost(model, input_tokens, output_tokens)
    await db.llm_usage_logs.insert_one(
        {
            "feature": feature,
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "estimated_cost": float(cost),
            "tenant_id": tenant_id,
            "created_at": datetime.now(timezone.utc),
        }
    )
