"""Supabase JWT verification (legacy HS256 secret or asymmetric JWKS keys)."""
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None = None


@lru_cache
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, cache_keys=True)


def decode_token(token: str, settings: Settings) -> dict:
    header = jwt.get_unverified_header(token)
    alg = header.get("alg", "")
    if alg == "HS256":
        if not settings.supabase_jwt_secret:
            raise jwt.InvalidTokenError("HS256 token but SUPABASE_JWT_SECRET is not configured")
        key = settings.supabase_jwt_secret
    elif alg in ("RS256", "ES256"):
        key = _jwks_client(settings.jwks_url).get_signing_key_from_jwt(token).key
    else:
        raise jwt.InvalidTokenError(f"Unsupported alg {alg}")
    return jwt.decode(token, key, algorithms=[alg], audience=settings.jwt_audience)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Falta el token de autenticación")
    try:
        claims = decode_token(creds.credentials, settings)
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Token inválido: {exc}") from exc
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token sin usuario")
    return CurrentUser(id=sub, email=claims.get("email"))
