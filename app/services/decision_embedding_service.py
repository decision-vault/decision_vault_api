from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import asyncpg

from app.core.config import settings
from app.db.postgres import get_pg_pool


@dataclass
class DecisionEmbeddingRecord:
    decision_id: str
    tenant_id: str
    project_id: str
    title: str
    statement: str | None
    context: str | None
    source_url: str | None
    embedding: list[float]


_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def generate_embedding(text: str) -> list[float]:
    model = _load_model()
    vector = model.encode([text], normalize_embeddings=True)
    return vector[0].tolist()


def _build_embedding_text(title: str | None, statement: str | None, context: str | None) -> str:
    parts = [p for p in [title, statement, context] if p]
    return "\n".join(parts).strip()


async def upsert_decision_embedding(record: DecisionEmbeddingRecord) -> None:
    pool = await get_pg_pool()
    await pool.execute(
        """
        INSERT INTO decision_embeddings (
            decision_id,
            tenant_id,
            project_id,
            title,
            statement,
            context,
            source_url,
            embedding
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        ON CONFLICT (decision_id) DO UPDATE SET
            tenant_id = EXCLUDED.tenant_id,
            project_id = EXCLUDED.project_id,
            title = EXCLUDED.title,
            statement = EXCLUDED.statement,
            context = EXCLUDED.context,
            source_url = EXCLUDED.source_url,
            embedding = EXCLUDED.embedding
        """,
        record.decision_id,
        record.tenant_id,
        record.project_id,
        record.title,
        record.statement,
        record.context,
        record.source_url,
        record.embedding,
    )


async def embed_and_store_decision(payload: dict) -> None:
    text = _build_embedding_text(payload.get("title"), payload.get("statement"), payload.get("context"))
    if not text:
        return
    vector = generate_embedding(text)
    record = DecisionEmbeddingRecord(
        decision_id=str(payload.get("_id") or payload.get("decision_id")),
        tenant_id=str(payload.get("tenant_id")),
        project_id=str(payload.get("project_id")),
        title=str(payload.get("title") or ""),
        statement=payload.get("statement"),
        context=payload.get("context"),
        source_url=payload.get("source_url"),
        embedding=vector,
    )
    await upsert_decision_embedding(record)


async def search_similar_decisions(
    tenant_id: str,
    project_id: str,
    query_embedding: Iterable[float],
    threshold: float,
    limit: int,
) -> list[dict]:
    pool: asyncpg.Pool = await get_pg_pool()
    rows = await pool.fetch(
        """
        SELECT decision_id, title, statement, context, source_url,
               1 - (embedding <=> $1) AS similarity
        FROM decision_embeddings
        WHERE tenant_id = $2 AND project_id = $3
          AND 1 - (embedding <=> $1) >= $4
        ORDER BY embedding <=> $1 ASC
        LIMIT $5
        """,
        list(query_embedding),
        tenant_id,
        project_id,
        threshold,
        limit,
    )
    results = []
    for row in rows:
        results.append(
            {
                "id": row["decision_id"],
                "title": row["title"],
                "excerpt": (row["statement"] or row["context"] or "")[:240],
                "similarity": float(row["similarity"]),
            }
        )
    return results
