CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS decision_embeddings (
    decision_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    statement TEXT,
    context TEXT,
    source_url TEXT,
    embedding VECTOR(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS decision_embeddings_tenant_project_idx
    ON decision_embeddings (tenant_id, project_id);

CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_idx
    ON decision_embeddings USING ivfflat (embedding vector_cosine_ops);
