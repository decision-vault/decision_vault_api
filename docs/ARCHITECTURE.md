# DecisionVault API — Internal Architecture & Design Notes

This document describes the `decision_vault_api` backend: package layout, major concepts, request pipeline, and the “why” behind the strategies implemented.

> Scope: `decision_vault_api/app/*` (FastAPI + MongoDB primary) plus optional Redis + optional Postgres integrations.

## 1) High-level overview

DecisionVault API is a multi-tenant FastAPI backend that provides:

- **Authentication**: password login + Google OAuth, short-lived JWT access tokens, refresh-token rotation.
- **Multi-tenancy**: each user belongs to exactly one tenant (“org”).
- **Authorization**: org-level roles + project-level roles (membership collection).
- **Licensing**: plan/status gating for product features (trial/starter/team/enterprise) with read-only modes.
- **Connectors**: Slack, Microsoft Teams, Zoom, Google Chat, and a **Custom Connector** ingestion API.
- **Product-generation pipelines**: requirements intake → clarifications → PRD/SDD/schema/usecase/sequence/architecture docs.
- **Operational controls**: idempotency, rate limiting (optional), caching (Redis with memory fallback), audit logging.

Entrypoint: `app/main.py` creates the FastAPI app, configures middleware, creates indexes on startup, and includes routers.

## 2) Package map (what lives where)

Top-level python package: `app/`

- `app/main.py`
  - FastAPI app, CORS + session middleware, startup index creation, router registration, health endpoints.
- `app/core/`
  - `config.py`: `Settings` (`DV_` env vars) and application config.
  - `rbac.py`: org/project roles and permission matrices.
  - `license.py`: feature → plan mapping and write-blocked features.
  - `errors.py`: app-specific exception types (e.g., `LicenseError`).
- `app/db/`
  - `mongo.py`: Motor client + database accessor.
  - `postgres.py`: optional `asyncpg` pool (used by some PRD/usage helpers).
- `app/middleware/`
  - `auth.py`: Bearer JWT access-token validation → sets `request.state.user`.
  - `tenant.py`: tenant resolution + “tenant mismatch” protection → sets `request.state.tenant_id`.
  - `rbac.py`: org/project authorization dependencies (reads membership from DB).
  - `license.py`: license status checks + feature gating.
  - `guard.py`: composable dependency `withGuard(...)` used across most routes.
- `app/api/`
  - FastAPI routers for auth, orgs, projects, connectors, PRD/requirements/generation, docs fetch, webhooks, etc.
- `app/services/`
  - “Business logic” modules (DB reads/writes, connector logic, generation orchestrators, caching, crypto, audit).
- `app/schemas/`
  - Pydantic request/response models (API boundary contracts).
- `app/utils/`
  - Shared helpers (JWT helpers, serialization, security, requirements parsing helpers, etc.).

### 2.1 Services index (quick reference)

Common “entry points” in `app/services/`:

- Auth & security: `auth_service.py`, `crypto_service.py`, `token_limiter.py`
- Licensing: `license_service.py`
- Audit: `audit_service.py`
- Projects & access: `project_service.py`, `project_member_service.py`, `project_access_service.py`
- Messaging: `messenger_service.py`
- Billing: `billing_service.py`, `stripe_webhook_service.py`
- Connectors:
  - Slack: `slack_service.py` (+ `slack_admin_service.py`)
  - Teams: `teams_service.py`, `teams_delta_service.py`, `teams_subscription_service.py`
  - Zoom: `zoom_service.py`
  - Google Chat: `google_chat_service.py`
  - Custom connector ingestion: `custom_connector_service.py`
- Generation:
  - Requirements intake + PRD-from-structured: `requirements_service.py`
  - PRD graph generation: `prd_graph_service.py` (+ `prd_multistep_service.py`)
  - Why query: `why_query_service.py` (+ `why_query_v2_service.py`)
  - System-design artifacts: `system_design_service.py`, `schema_flow_service.py`, `usecase_flow_service.py`, `architecture_mermaid_service.py`
- Infra helpers: `cache_service.py`, `email_service.py`

## 3) Configuration model (and why)

`app/core/config.py` defines a `pydantic-settings` `Settings` class loaded from `DV_*` env vars.

Why this approach:
- **Single source of truth** for operational parameters (token TTLs, connector keys, rate-limiter toggles).
- **Local-dev friendly defaults** while still allowing production overrides via environment variables.
- Makes “optional dependencies” explicit (Redis, Postgres, OAuth) without hard-failing the app startup.

Important settings families:
- Auth/JWT: `DV_JWT_SECRET`, `DV_ACCESS_TOKEN_MINUTES`, `DV_REFRESH_TOKEN_DAYS`, issuer/audience.
- Cookies: `DV_SECURE_COOKIES`, `DV_COOKIE_SAMESITE`, `DV_COOKIE_DOMAIN`.
- Connectors: `DV_SLACK_*`, `DV_TEAMS_*`, `DV_ZOOM_*`, `DV_GOOGLE_CHAT_*`, `DV_CUSTOM_CONNECTOR_*`.
- Infra toggles: `DV_ENABLE_RATE_LIMITER`, `DV_REDIS_URL`, `DV_POSTGRES_DSN`.
- LLM: provider selection + base URLs + token budgets.

## 3.1 Dependency map (requirements files → purpose)

DecisionVault pins dependencies in:

- `requirements.txt` (local/dev + tests + optional local model runtime)
- `requirements.cloud.txt` (cloud/runtime focused; omits dev-only and heavy local-only deps)

Below is what each dependency is used for in this codebase.

### Web framework & server

- `fastapi`: HTTP API framework (routers, dependencies, request lifecycle).
- `uvicorn`: ASGI server used to run FastAPI in dev/prod.

### MongoDB driver stack

- `motor`: async MongoDB driver used throughout `app/db/mongo.py` + `app/services/*`.
- `pymongo`: underlying Mongo driver; also used for helpers like `ReturnDocument`.

### Auth, sessions, and security primitives

- `python-jose`: JWT encode/decode in `app/utils/token.py`.
- `passlib[bcrypt]` + `bcrypt`: password hashing/verification in `app/utils/security.py` and user auth flows.
- `authlib`: OAuth client for Google login in `app/api/auth.py`.
- `pydantic-settings`: typed config via env vars in `app/core/config.py`.
- `itsdangerous`: used by Starlette/FastAPI session signing (via `SessionMiddleware` in `app/main.py`).
- `email-validator`: validates invite/signup email inputs via Pydantic schemas.
- `cryptography`: encryption building blocks used by `app/services/crypto_service.py` for connector token storage.

### Billing / payments

- `stripe`: Stripe webhook verification + subscription event handling in `app/api/webhooks.py` and `app/services/stripe_webhook_service.py`.
- (Razorpay is used via direct HTTP calls) `httpx`: used for Razorpay payment link creation in `app/services/billing_service.py`.

### HTTP client utilities

- `httpx`: used across connectors (Slack OAuth exchange, external APIs) and LLM provider calls (LM Studio/HF router paths).

### Rate limiting & caching

- `fastapi-limiter`: request-rate limiting (Redis-backed) used for `demo` and `custom` ingestion routes.
- `redis`: async Redis client used by `fastapi-limiter` and `app/services/cache_service.py`.

### Multipart uploads

- `python-multipart`: enables `request.form()` / form parsing used in custom connector OAuth token endpoint.

### LLM orchestration & graph pipelines

- `langchain`: prompt + output parsing primitives used in PRD/why-query flows.
- `langgraph`: explicit deterministic control flow graphs for requirements + PRD generation + why-query (`StateGraph` patterns).
- `langchain-openai`: ChatOpenAI wrapper used for OpenAI-compatible endpoints and LM Studio/OpenAI-style APIs.
- `langchain-google-genai` + `google-generativeai`: Gemini provider support in some LLM flows (provider switch in settings).
- `openai`: OpenAI SDK used by some LLM paths (and/or OpenAI-compatible provider clients).

### Optional Postgres + vectors (not always enabled)

- `asyncpg`: optional Postgres connection pool in `app/db/postgres.py` (and PRD/usage helpers).
- `pgvector`: planned/optional vector search support; referenced as a future stage in why-query v2 / vector memory areas.

### Local model runtime (optional, dev/heavy)

- `transformers`: local HF model inference mode in `app/api/hf_inference.py` + `app/services/hf_model_loader.py`.
- `huggingface_hub`: remote/local model loading and model metadata access for HF flows.
- `torch`: required for local Transformers inference (present in `requirements.txt`, omitted in `requirements.cloud.txt`).

### Testing (dev only)

- `pytest`: test runner.
- `pytest-asyncio`: async test support for FastAPI/service coroutines.

## 4) Data stores (and why two of them exist)

### MongoDB (primary)
Used for:
- Tenants, users, refresh tokens
- Licenses + audit logs
- Projects, project memberships/access requests
- Connectors installations + webhook idempotency logs
- Generated docs (SDD/schema/usecase/sequence/architecture) and run tracking
- Messaging (“project messenger”) collections

Why MongoDB:
- Flexible schema for connector payloads and generated artifacts.
- Easy incremental evolution for “document-like” data (generated markdown, graph nodes/edges, etc.).
- Fast indexing patterns for multi-tenant lookups.

### Redis (optional)
Used for:
- `fastapi-limiter` rate limiting (when enabled and Redis is reachable).
- Application caching (`app/services/cache_service.py`) where Redis is preferred but falls back to in-memory.

Why Redis as optional:
- You still want local dev / minimal deployments to work without Redis.
- Rate limiting is a defense-in-depth control; the app degrades gracefully if Redis is unavailable.

### Postgres (optional)
Used for:
- Some PRD-related helpers / usage tracking paths (startup calls `ensure_prd_table()` / `ensure_usage_table()` are guarded and log warnings if Postgres isn’t configured).

Why Postgres is optional:
- Some “analytics / structured storage” use-cases are better in relational stores, but the app must run without it.

## 5) Request pipeline (from HTTP → authorization → business logic)

DecisionVault uses **FastAPI dependencies** as its primary “middleware/pipeline” mechanism.

### 5.1 Built-in middleware
Configured in `app/main.py`:
- **CORS middleware**: allows localhost + common preview hosts (ngrok/vercel) with credentials enabled.
- **Session middleware**: used for Google OAuth callback flow (`/api/auth/google`).

### 5.2 Per-route dependency pipeline: `withGuard(...)`
Most “real” routes use:

`withGuard(feature=..., orgRole=..., projectRole=...)` from `app/middleware/guard.py`.

It composes these checks in-order:
1. `get_current_user` (`app/middleware/auth.py`)
   - Validates Bearer token, requires `type=access`, verifies JWT, loads user doc, checks `tenant_id` match.
   - Stores `request.state.user = {user_id, tenant_id, role}`.
2. `resolve_tenant` (`app/middleware/tenant.py`)
   - Resolves tenant from path/query/header or uses user’s tenant.
   - Enforces “tenant mismatch” protection, stores `request.state.tenant_id`.
3. `assertLicense(feature)` (`app/middleware/license.py`)
   - Loads license doc and computes status (active/expired/grace/suspended).
   - Enforces plan requirements and write-blocking in expired/grace.
   - Stores `request.state.license` and `request.state.license_status`.
4. Optional `requireOrgRole(...)` and/or `requireProjectRole(...)` (`app/middleware/rbac.py`)
   - Org roles are derived from the user’s role claim.
   - Project roles are derived from `project_members` collection.

Why dependencies (vs global middleware):
- Each route can express the *exact* requirements (feature gate + org role + project role).
- Avoids hidden global authorization rules; the “policy” is visible near each endpoint.
- Keeps auth/rbac/license logic unit-testable as separate functions.

### 5.3 Rate limiting (selected routes)
Two patterns exist:
- Global init: `FastAPILimiter.init(redis_client)` is called on startup **only if enabled** and Redis is reachable.
- Per-route limiter: e.g. `/api/demo/requests` and `/api/custom/ingest` call a small helper dependency that no-ops if Redis/limiter isn’t available.

Why per-route:
- Focuses limiting on externally-facing or abuse-prone endpoints (lead forms, ingestion endpoints).

## 6) Identity & security concepts

### 6.1 Access + refresh token strategy
Files: `app/services/auth_service.py`, `app/utils/token.py`, `app/api/auth.py`

- Access tokens are **JWT bearer tokens** returned in JSON (`expires_in` returned too).
- Refresh tokens are **JWT** stored in a **HttpOnly cookie** (`dv_refresh`) scoped to `/api/auth`.
- Refresh tokens are persisted server-side in `refresh_tokens` with:
  - `jti` unique index for lookup/idempotency.
  - `token_hash` (HMAC using `DV_JWT_SECRET`) to detect token theft/reuse.
  - `revoked` + `replaced_by` for rotation chains.

Refresh rotation + reuse detection (`auth_service.refresh`):
- If token doc is missing/revoked → revoke all user refresh tokens (defense-in-depth) and deny.
- If hash mismatch → treat as **reuse** (stolen/duplicated token) → revoke all user refresh tokens and deny.
- Otherwise issue new tokens, store new refresh token, revoke old token with `replaced_by`.

Why this design:
- Keeps access tokens short-lived while enabling sessions without persistent server-side sessions.
- Rotation limits damage from refresh-token theft and gives an explicit “reuse detection” signal.
- HttpOnly cookie reduces XSS exposure vs storing refresh tokens in JS-accessible storage.

### 6.2 Multi-tenant isolation
Every authenticated request ties to one tenant:
- Token includes `tenant_id`.
- `get_current_user` ensures DB user’s tenant matches token’s `tenant_id`.
- `resolve_tenant` ensures requested tenant (path/query/header) matches user’s tenant.

Why both checks exist:
- Token claim is not enough (user could be deleted, deactivated, or moved).
- Request-scoped tenant checks prevent “horizontal privilege escalation” across tenants.

### 6.3 RBAC model
`app/core/rbac.py` defines:
- Org roles (viewer/member/admin/owner) and permission matrix.
- Project roles (viewer/contributor/project_admin) and permission matrix.
- Product-wide “superAdmin” bypass role.

Project authorization is membership-based:
- Membership doc is looked up in `project_members` by `(tenant_id, project_id, user_id)` plus `deleted_at=None`.

Why split org vs project roles:
- Org ownership (billing/integrations/admin operations) should not automatically grant project write access.
- Project membership enables fine-grained collaboration across many projects inside the same tenant.

### 6.4 Licensing gates
Files: `app/core/license.py`, `app/services/license_service.py`, `app/middleware/license.py`

The API treats licensing as a “feature gate” + “write blocking” mechanism:
- Some features are always allowed (e.g., `view_decision`, `search`).
- For expired/grace, the system enforces read-only for write-like features.
- Plan requirements are checked for features like integrations management.

Why do this at request time:
- Keeps licensing enforcement consistent across all clients (UI, mobile, connectors).
- Prevents connector ingestion from bypassing UI-level read-only restrictions.

## 7) API surface overview (router-by-router)

> Many routers are “thin controllers” that call into `app/services/*`.

### Auth (`app/api/auth.py`)
- `/api/auth/signup`, `/login`, `/refresh`, `/logout`
- Google OAuth start/callback
- `/api/auth/session` returns user+tenant metadata for the UI.

### Orgs (“Tenants”) (`app/api/orgs.py`)
- `/api/orgs/me` (read/update/delete current org)
- Invite flow: create invite + email send in background + accept flow in services
- Super-admin org list/create (restricted)

### Projects (`app/api/projects.py`)
- Lists projects by membership (joins via `project_members`)
- Catalog endpoints + access request workflow
- A “route collision avoidance” strategy is used: static paths like `/catalog` have `/meta/*` or `/access/*` variants to avoid ambiguous matching in older deployments.

### Resources (`app/api/resources.py`)
Aggregated “utility” endpoints:
- License CRUD + banner
- Project member management
- Project catalog + access requests (some overlap with `projects.py`)
- LLM health probe (`probe_llm`)

### Messenger (`app/api/messenger.py`)
Project-scoped internal messaging:
- Channels, threads, messages
- Personal chats + messages
- Favorites

### Requirements (`app/api/requirements.py`)
Implements the requirements intake → clarification chat → downstream doc generation runs.

Key concepts:
- **Chat message normalization** to keep the UI in a one-question-at-a-time flow (backward compatibility helpers).
- **Run tracking** documents (per stage) for async multi-step generation.
- Background async tasks are tracked by `_ACTIVE_*_TASKS_BY_RUN_ID` maps.

### PRD (`app/api/prd.py`)
PRD generation and export:
- Multi-step PRD generation stages (`PRD_STAGE_SEQUENCE`) with pause/resume/stop/retry semantics.
- Token budgeting (`TokenLimiter`) to prevent runaway prompts.
- Optional PDF export pipeline (custom markdown → PDF bytes; includes header/footer/watermark).

### Generation aggregator (`app/api/generation.py`)
Provides a unified API for starting and controlling runs:
- `kind ∈ {prd,sdd,schema,usecase,sequence,architecture}`
- Stores a single `generation_runs` document that references a “child” run (`child_run_id`) created by underlying modules.
- Supports “regenerate” semantics by stopping active runs of the same kind and starting a new run.

Why this layer exists:
- Keeps the UI simple: one “runs” concept, even though multiple generation subsystems exist underneath.
- Enables uniform pause/resume/stop/status and a place to attach doc IDs.

### Connectors status (`app/api/connectors.py`)
Unified “connected/not connected” view + start URL + disconnect flows (Slack currently).

### Slack connector (`app/api/slack_connector.py`)
Full Slack OAuth + event ingestion:
- OAuth start/callback: stores encrypted bot token (`app/services/crypto_service.py` + `slack_service`).
- Signature verification (timestamp window for replay protection).
- Scoping: allowed channels list.
- Ingestion strategy: “high precision capture”
  - Only capture if channel is allowed, license allows capture, message is threaded, and text contains a decision signal.
- Background processing via `BackgroundTasks`.

### Teams connector (`app/api/teams_connector.py`)
Microsoft Graph-based integration:
- OAuth + subscription webhook endpoint
- Scoping: team/channel IDs and allow/private toggle
- “Delta sync” support for historical capture

### Zoom connector (`app/api/zoom_connector.py`)
Zoom OAuth + webhook verification + idempotency.

### Google Chat connector (`app/api/google_chat_connector.py`)
Install-by-domain model + webhook signature/token verification + per-space scoping + thread activity tracking.

### Custom connector (`app/api/custom_connector.py`)
External ingestion endpoint with multiple auth options:
- Auth mechanisms (mutually exclusive):
  - Bearer OAuth token (client credentials)
  - `x-api-key` (rotatable, hashed at rest)
  - `x-signature` HMAC (shared secret)
- Payload-size enforcement via `Content-Length`
- Idempotency by `(tenant_id, external_id)` using `custom_connector_requests` collection
- Delivery logging + retry queue with exponential backoff
- Optional Redis-based rate limiting on the ingest route

Why multiple auth options:
- Enables adoption across many client types (server-to-server, webhook relays, constrained environments).
- Supports gradual upgrade from shared secrets → rotated keys → OAuth client credentials.

### Webhooks (`app/api/webhooks.py`)
Stripe webhook handler with signature verification and idempotent processing (via `stripe_events` collection).

### Docs fetch (`app/api/docs.py`)
Unified “doc by id” retrieval for multiple document kinds.
Why it exists:
- The UI can open any generated artifact through a single endpoint and render markdown/mermaid consistently.

### HF Inference (`app/api/hf_inference.py`)
Model inference endpoint supporting:
- Remote mode (LM Studio / OpenAI-style client)
- Local mode (Transformers model)
- Prompt truncation by tokens/chars to guard model limits

## 8) Generation pipelines (concepts + strategies)

### 8.1 Requirements intake → clarification graph (LangGraph)
File: `app/services/requirements_service.py`

Flow:
1. Parse raw text into a partial structured dict (regex + heuristics).
2. Validate “required fields” + “low quality fields”.
3. Build clarification questions for missing/low-quality fields.
4. Collect answers and deep-merge into the structured model, then re-validate.

Why this approach:
- Keeps a deterministic loop: “validate → ask only what’s missing”.
- Avoids a single LLM call that might invent details.
- Produces a well-defined structured object that downstream generators can trust.

Quality strategy:
- Fields are classified as descriptive vs enum vs lists.
- Placeholder detection prevents “TBD/later” from satisfying requirements.

### 8.2 PRD generation graph (anti-hallucination + token budgets)
File: `app/services/prd_graph_service.py`

Key strategies:
- **Structured output parsing** (PydanticOutputParser) to force JSON shape.
- **Provider abstraction** supports `huggingface`, `lmstudio`, or OpenAI-style endpoints.
- **TokenLimiter** enforces `DV_LLM_MAX_INPUT_TOKENS` and `DV_LLM_MAX_OUTPUT_TOKENS`.
- **Hallucination detection**:
  - Flags new numbers/integrations/currencies/percentages not present in the input.
  - If flagged, retries with a stricter instruction to remove invented details.
- **Caching**:
  - Per-tenant cache key computed from normalized input (Redis preferred, memory fallback).

Why this approach:
- PRDs are high-risk for “confidently wrong” output; the pipeline nudges toward faithful expansion.
- Token budgets prevent latency spikes and cost blowups.
- Caching reduces repeated generation for identical inputs.

### 8.3 “Unified runs” orchestration
File: `app/api/generation.py`

Strategy:
- One `generation_runs` record per (project, kind) run, storing `status`, `child_run_id`, timestamps, and errors.
- Supports replace/regenerate by stopping prior active runs best-effort.

Why:
- Keeps UI behavior consistent across PRD/SDD/schema/usecase/sequence/architecture runs.
- Enables the backend to evolve individual generators without changing the UI contract.

## 9) Operational patterns implemented (and why)

### Index-first startup
`app/main.py` creates indexes for critical collections and includes safety checks:
- Skips unique indexes when duplicates exist (boot safety for existing environments).
- Logs structured warnings describing what migration is needed.

Why:
- Multi-tenant uniqueness constraints are essential for correctness (e.g., per-tenant email uniqueness).
- Skipping on duplicates prevents the whole service from failing to start in “dirty” environments.

### Idempotency
Used for:
- Stripe webhooks (`stripe_events`)
- Custom connector ingestion (`custom_connector_requests` keyed by `external_id`)
- Zoom webhook event dedupe (service-level helper)

Why:
- Webhooks and external ingestion are retried by providers; idempotency prevents duplicate decisions/billing events.

### Encryption at rest for connector tokens
Slack bot token is stored encrypted (`app/services/crypto_service.py` is used by `slack_service`).

Why:
- Limits blast radius if MongoDB is leaked/read-only compromised.

### Audit logging
`app/services/audit_service.py` logs security-sensitive actions (signup/login, license changes, invite workflows, etc.).

Why:
- Essential for debugging tenant issues and tracking administrative actions.

## 10) Testing

Tests live in `decision_vault_api/tests/`.
The codebase uses `pytest` + `pytest-asyncio` for async test support.

## 11) “Important notes” for maintainers

- Several endpoints under `app/api/` are intentionally minimal placeholders (e.g., `example.py`, `decisions.py`, `uploads.py`, parts of `slack.py`); most business logic is implemented in `services/` and the “connector” routers.
- Many config values in `Settings` have non-empty defaults; treat them as **development placeholders** and override via environment variables in production.
- Redis and Postgres are treated as optional dependencies; the service should still start and serve core API functionality without them.
