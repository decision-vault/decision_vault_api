# DecisionVault Backend

This backend provides secure, multi-tenant authentication for DecisionVault using FastAPI, MongoDB, JWT access tokens, refresh token rotation, and Google OAuth.

## Requirements

- Python 3.14+
- MongoDB

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (example):

```bash
export DV_MONGO_URI="mongodb://localhost:27017"
export DV_MONGO_DB="decisionvault"
export DV_JWT_SECRET="replace-with-strong-secret"
export DV_SESSION_SECRET="replace-with-strong-secret"
export DV_GOOGLE_CLIENT_ID="your-google-client-id"
export DV_GOOGLE_CLIENT_SECRET="your-google-client-secret"
export DV_GOOGLE_REDIRECT_URI="http://localhost:8000/api/auth/google/callback"
export DV_FRONTEND_BASE_URL="http://localhost:3000"
export DV_SECURE_COOKIES="false"
```

## Run

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Auth Endpoints

- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/refresh`
- `POST /api/auth/logout`
- `GET /api/auth/google`
- `GET /api/auth/google/callback`

Refresh tokens are stored in a secure, HttpOnly cookie (`dv_refresh`). Access tokens are returned in JSON.

## Notes

- Passwords are hashed with bcrypt (cost factor 12).
- JWT access tokens expire after 15 minutes.
- Refresh tokens expire after 7 days and are rotated on use.
- Each user belongs to exactly one tenant.
- Email uniqueness is enforced per tenant.
