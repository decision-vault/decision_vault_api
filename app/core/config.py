from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DecisionVault"
    mongo_uri: str = "mongodb://localhost:27017"
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
    google_redirect_uri: str = "http://localhost:8000/api/auth/google/callback"

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
    ]
    cors_allow_methods: list[str] = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    cors_allow_headers: list[str] = ["*"]

    secure_cookies: bool = True
    cookie_samesite: str = "none"
    cookie_domain: str | None = None

    session_secret: str = "change-me"

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_starter: str = ""
    stripe_price_team: str = ""
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_currency: str = "INR"
    razorpay_amount_starter_paise: int = 0
    razorpay_amount_team_paise: int = 0

    slack_client_id: str = "10553321666597.10560383537988"
    slack_client_secret: str = "cb7d74cc9d121c43e621ee51060fe2b5"
    slack_signing_secret: str = "5ab2b0c892a56ee904878775de2ccca0"
    slack_redirect_uri: str = "https://homoeomorphic-especially-felecia.ngrok-free.dev/api/slack/oauth/callback"
    slack_token_encryption_key: str = "uoZWtkpkDu71zaMpSzHnzpso"
    slack_channel_cache_seconds: int = 300

    teams_client_id: str = ""
    teams_client_secret: str = ""
    teams_tenant_id: str = ""
    teams_redirect_uri: str = "http://localhost:8000/api/teams/oauth/callback"
    teams_token_encryption_key: str = ""

    zoom_client_id: str = ""
    zoom_client_secret: str = ""
    zoom_redirect_uri: str = "http://localhost:8000/api/zoom/oauth/callback"
    zoom_token_encryption_key: str = ""
    zoom_webhook_secret: str = ""
    zoom_channel_cache_seconds: int = 300

    google_chat_service_account_json: str = ""
    google_chat_delegated_user: str = ""
    google_chat_webhook_token: str = ""
    google_chat_project_id: str = ""
    google_chat_webhook_secret: str = ""
    google_chat_space_cache_seconds: int = 300

    custom_connector_hmac_secret: str = ""
    custom_connector_rate_limit: str = "10/minute"
    custom_connector_max_payload_bytes: int = 262144
    custom_connector_oauth_token_minutes: int = 60
    custom_connector_oauth_audience: str = "decisionvault-custom"
    custom_connector_retry_base_seconds: int = 30
    custom_connector_retry_max_seconds: int = 3600
    custom_connector_retry_max_attempts: int = 5
    redis_url: str = "redis://default:ONaXQZ73sLktXIOm8qkAseLRkzgvo9B6@redis-15917.c10.us-east-1-4.ec2.cloud.redislabs.com:15917"
    enable_rate_limiter: bool = True

    postgres_dsn: str = ""
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_search_enabled: bool = False
    llm_provider: str = "lmstudio"
    llm_model: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    llm_temperature: float = 0.2
    llm_api_key: str = "AIzaSyDgU2vgycUahBqvViPh9oFezpuD7L9GAgM"
    llm_base_url: str | None = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_max_input_tokens: int = 3000
    llm_max_output_tokens: int = 1800
    prd_generation_timeout_seconds: int = 180

    hf_api_token: str = "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"
    hf_model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    hf_tokenizer_name: str = ""
    hf_router_base_url: str = "https://router.huggingface.co/v1"
    hf_openai_model: str = "Qwen/Qwen2.5-7B-Instruct:together"
    lmstudio_base_url: str = "http://localhost:1234/api/v1"
    lmstudio_model: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    remote_provider: str = "lmstudio"
    lmstudio_chat_path: str = "/chat"
    model_run_mode: str = "remote"
    local_model_path: str = ""
    local_device: str = ""
    hf_system_prompt: str = "You are a helpful assistant."
    hf_max_input_tokens: int = 2048
    hf_max_input_chars: int = 8000

    class Config:
        env_prefix = "DV_"
        protected_namespaces = ("settings_",)


settings = Settings()
