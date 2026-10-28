"""Passwords (argon2id), access tokens (short-lived JWTs) and refresh tokens (opaque, hashed)."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.config import DEV_JWT_SECRET, Settings

_hasher = PasswordHasher()  # argon2id with the library's current recommended parameters
_ALGORITHM = "HS256"
_ACCESS = "access"
# Verified against when the user does not exist, so a login attempt takes the same time either
# way and response timing does not reveal which email addresses have accounts.
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")

MIN_PASSWORD_LENGTH = 12


class InvalidTokenError(Exception):
    pass


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def check_secret(settings: Settings) -> None:
    secret = settings.jwt_secret.get_secret_value()
    if settings.is_prod and (secret == DEV_JWT_SECRET or len(secret) < 32):
        raise RuntimeError("JWT_SECRET must be set to a random value of 32+ characters.")


def _secret(settings: Settings) -> str:
    check_secret(settings)
    return settings.jwt_secret.get_secret_value()


@dataclass(frozen=True)
class AccessClaims:
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    role: str


def create_access_token(
    settings: Settings, *, user_id: uuid.UUID, tenant_id: uuid.UUID, role: str
) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "tid": str(tenant_id),
        "role": role,
        "type": _ACCESS,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, _secret(settings), algorithm=_ALGORITHM)


def decode_access_token(settings: Settings, token: str) -> AccessClaims:
    try:
        payload = jwt.decode(
            token,
            _secret(settings),
            algorithms=[_ALGORITHM],  # pinned: never let the token choose its algorithm
            options={"require": ["sub", "tid", "exp", "type"]},
        )
        if payload["type"] != _ACCESS:
            raise InvalidTokenError("not an access token")
        return AccessClaims(uuid.UUID(payload["sub"]), uuid.UUID(payload["tid"]), payload["role"])
    except (jwt.PyJWTError, ValueError, KeyError) as exc:
        raise InvalidTokenError(str(exc)) from exc


def new_refresh_token() -> tuple[str, str]:
    """(token for the client, SHA-256 hash to store). The token itself is never stored."""
    token = secrets.token_urlsafe(48)
    return token, hash_refresh_token(token)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
