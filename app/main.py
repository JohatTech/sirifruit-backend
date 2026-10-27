import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import predict, references
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sirifruit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Tests may pre-populate app.state with fakes.
    if not hasattr(app.state, "embedder"):
        from app.ml.embedder import MobileNetEmbedder

        log.info("Cargando MobileNetV3-Small…")
        app.state.embedder = MobileNetEmbedder(threads=settings.torch_threads)
    if not hasattr(app.state, "store"):
        from app.services.reference_store import SupabaseReferenceStore

        if not settings.supabase_url or not settings.supabase_service_key:
            raise RuntimeError("Configura SUPABASE_URL y SUPABASE_SERVICE_KEY en las variables de entorno de Render / .env")
        app.state.store = SupabaseReferenceStore(
            settings.supabase_url, settings.supabase_service_key, settings.storage_bucket
        )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="SiriFruit API", version="1.0.0", lifespan=lifespan)
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(predict.router)
    app.include_router(references.router)

    @app.get("/health", tags=["sistema"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
