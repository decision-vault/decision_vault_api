import base64
import hashlib

from cryptography.fernet import Fernet

from app.core.config import settings


def _fernet(key: str) -> Fernet:
    raw = key.encode("utf-8")
    try:
        return Fernet(raw)
    except Exception:
        # Dev-friendly fallback: derive a valid Fernet key from any input string.
        derived = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
        return Fernet(derived)


def encrypt_token(value: str) -> str:
    if not settings.slack_token_encryption_key:
        raise ValueError("Missing slack token encryption key")
    return _fernet(settings.slack_token_encryption_key).encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_token(value: str) -> str:
    if not settings.slack_token_encryption_key:
        raise ValueError("Missing slack token encryption key")
    return _fernet(settings.slack_token_encryption_key).decrypt(value.encode("utf-8")).decode("utf-8")


def encrypt_token_with_key(value: str, key: str) -> str:
    return _fernet(key).encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_token_with_key(value: str, key: str) -> str:
    return _fernet(key).decrypt(value.encode("utf-8")).decode("utf-8")
