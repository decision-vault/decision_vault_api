from passlib.hash import bcrypt

from app.core.config import settings


def hash_password(password: str) -> str:
    return bcrypt.using(rounds=settings.bcrypt_cost).hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.verify(password, hashed)
