from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    supabase_url: str = ""
    # Accepts the new secret key (sb_secret_...) or the legacy service_role JWT.
    supabase_service_key: str = Field(
        default="",
        validation_alias=AliasChoices("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"),
    )
    supabase_jwks_url: str = ""
    # Legacy HS256 secret. If empty, tokens are verified with the project's JWKS (asymmetric keys).
    supabase_jwt_secret: str = ""
    jwt_audience: str = "authenticated"

    storage_bucket: str = "pack-photos"

    # Recognition
    conf_threshold: float = 0.6       # min confidence of the top flavor
    margin_threshold: float = 0.25    # min gap between top-1 and top-2 confidence
    embedding_weight: float = 0.7     # weight of CNN similarity vs color similarity
    top_k_per_flavor: int = 3
    softmax_temperature: float = 0.05

    max_upload_mb: float = 5.0
    cors_origins: str = "*"
    torch_threads: int = 2

    @property
    def jwks_url(self) -> str:
        return self.supabase_jwks_url or f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
