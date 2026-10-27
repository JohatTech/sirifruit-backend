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
        if not settings.supabase_url or not settings.supabase_service_key:
            log.warning(
                "AVISO: SUPABASE_URL o SUPABASE_SERVICE_KEY no estan configuradas en Render. "
                "Agrega estas variables en el menu 'Environment' de Render para habilitar la persistencia."
            )
            from app.services.reference_store import UnconfiguredReferenceStore

            app.state.store = UnconfiguredReferenceStore()
        else:
            from app.services.reference_store import SupabaseReferenceStore

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
        settings = get_settings()
        return {
            "status": "ok",
            "supabase_configured": bool(settings.supabase_url and settings.supabase_service_key),
        }

    return app


app = create_app()
