from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DecisionVault"
    mongo_uri: str = "mongodb+srv://kaviyarasumaran:4jCFpJON76UbxyfK@cluster0.g5hin.mongodb.net/"
    mongo_db: str = "decisionvault"

    jwt_secret: str = "change-me"
    jwt_issuer: str = "decisionvault"
    jwt_audience: str = "decisionvault-users"

    access_token_minutes: int = 15
    refresh_token_days: int = 7

    trial_days: int = 14
    trial_grace_days: int = 7

    bcrypt_cost: int = 12

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "https://decision-vault-2gmw4vff9-kaviyarasumarans-projects.vercel.app/api/auth/google/callback"
    langgraph_url: str = "http://localhost:8050"  
    frontend_base_url: str = "http://localhost:3000"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:8081",
        "https://decision-vault-api-ujco.vercel.app",
        "https://decision-vault-2gmw4vff9-kaviyarasumarans-projects.vercel.app",
        "https://decision-vault-api-ujco.vercel.app",
        "https://decision-vault-api.vercel.app",
    ]
    cors_allow_methods: list[str] = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    cors_allow_headers: list[str] = ["*"]

    secure_cookies: bool = True
    cookie_samesite: str = "none"
    cookie_domain: str | None = None

    session_secret: str = "change-me"

    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_currency: str = "INR"
    razorpay_amount_starter_paise: int = 0
    razorpay_amount_team_paise: int = 0

    redis_url: str = "redis://localhost:6379/0"
    enable_rate_limiter: bool = True

    # When False (serverless hosts like Vercel), the interactive PRD chat step runs
    # synchronously instead of via in-process BackgroundTasks, and job state is
    # persisted to MongoDB so the polling endpoint works across invocations.
    background_jobs_enabled: bool = True

    #  FIX: Added missing dynamic cleanup routing target key
    doc_service_internal_url: str = "http://localhost:8000"

    postgres_dsn: str = ""
    llm_provider: str = "lmstudio"
    llm_model: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    llm_temperature: float = 0.7

    # Email (SMTP) - used for org/team invites
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = "decisionvaultai@gmail.com"
    smtp_password: str = "esft vdxa ifkk obsc"
    smtp_from_email: str = "decisionvaultai@gmail.com"
    smtp_from_name: str = "DecisionVault"
    smtp_use_starttls: bool = True

    org_invite_expires_hours: int = 72
    org_invite_frontend_path: str = "/invite"

    class Config:
        env_prefix = "DV_"
        protected_namespaces = ("settings_",)


settings = Settings()