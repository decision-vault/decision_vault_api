from app.core.config import settings


def get_hf_config() -> dict:
    """
    Environment-driven configuration:
    - HF_API_TOKEN: Hugging Face token for remote inference.
    - HF_MODEL_NAME: Model id (used for both remote and local by default).
    - MODEL_RUN_MODE: 'remote' or 'local'.
    - LOCAL_MODEL_PATH: Optional local model path override.
    """
    return {
        "hf_api_token": settings.hf_api_token,
        "hf_model_name": settings.hf_model_name,
        "hf_tokenizer_name": settings.hf_tokenizer_name,
        "model_run_mode": settings.model_run_mode,
        "local_model_path": settings.local_model_path,
        "local_device": settings.local_device,
        "hf_max_input_tokens": settings.hf_max_input_tokens,
        "hf_max_input_chars": settings.hf_max_input_chars,
    }
