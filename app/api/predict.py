from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_embedder, get_store, image_upload
from app.api.schemas import PredictionOut, PredictResponse
from app.core.auth import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.ml.embedder import Embedder
from app.ml.matcher import match
from app.ml.preprocess import InvalidImageError
from app.services.features import extract_features
from app.services.reference_store import ReferenceStore

router = APIRouter(tags=["reconocimiento"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    data: bytes = Depends(image_upload),
    user: CurrentUser = Depends(get_current_user),
    embedder: Embedder = Depends(get_embedder),
    store: ReferenceStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    try:
        emb, hist = await run_in_threadpool(extract_features, data, embedder)
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc)) from exc

    catalog = await run_in_threadpool(store.catalog, user.id)
    result = match(
        emb,
        hist,
        catalog.references,
        catalog.flavors,
        embedding_weight=settings.embedding_weight,
        top_k=settings.top_k_per_flavor,
        temperature=settings.softmax_temperature,
        conf_threshold=settings.conf_threshold,
        margin_threshold=settings.margin_threshold,
    )
    message = None
    if result.references_used == 0:
        message = "Aún no hay fotos de referencia. Entrena los sabores en la pestaña Sabores."
    elif result.low_confidence:
        message = "No estoy seguro, elige el sabor."

    return PredictResponse(
        predictions=[
            PredictionOut(flavor_id=p.flavor_id, slug=p.slug, name=p.name, confidence=p.confidence)
            for p in result.predictions
        ],
        low_confidence=result.low_confidence,
        references_used=result.references_used,
        message=message,
    )
