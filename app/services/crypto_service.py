from cryptography.fernet import Fernet

from app.core.config import settings


def _fernet(key: str) -> Fernet:
    return Fernet(key.encode("utf-8"))


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
