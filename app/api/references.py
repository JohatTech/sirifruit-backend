from collections import Counter
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_embedder, get_store, read_image
from app.api.schemas import FlavorStats, ReferenceCreated, ReferenceStatsResponse
from app.core.auth import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.ml.embedder import Embedder
from app.ml.preprocess import InvalidImageError
from app.services.features import extract_features
from app.services.reference_store import ReferenceStore

router = APIRouter(prefix="/references", tags=["referencias"])

MAX_FILES_PER_REQUEST = 20


@router.post("", response_model=ReferenceCreated, status_code=status.HTTP_201_CREATED)
async def add_references(
    flavor_id: str = Form(...),
    source: Literal["enroll", "correction"] = Form("enroll"),
    files: list[UploadFile] = File(...),
    user: CurrentUser = Depends(get_current_user),
    embedder: Embedder = Depends(get_embedder),
    store: ReferenceStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ReferenceCreated:
    if not files or len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Envía entre 1 y {MAX_FILES_PER_REQUEST} fotos")

    catalog = await run_in_threadpool(store.catalog, user.id)
    if flavor_id not in {f.id for f in catalog.flavors}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sabor no encontrado")

    ids: list[str] = []
    for f in files:
        data = await read_image(f, settings)
        try:
            emb, hist = await run_in_threadpool(extract_features, data, embedder)
        except InvalidImageError as exc:
            raise HTTPException(422, f"{f.filename}: {exc}") from exc
        rid = await run_in_threadpool(store.add_reference, user.id, flavor_id, emb, hist, data, source)
        ids.append(rid)

    catalog = await run_in_threadpool(store.catalog, user.id)
    total = sum(1 for r in catalog.references if r.flavor_id == flavor_id)
    return ReferenceCreated(ids=ids, flavor_id=flavor_id, total_for_flavor=total)


@router.get("/stats", response_model=ReferenceStatsResponse)
async def reference_stats(
    user: CurrentUser = Depends(get_current_user),
    store: ReferenceStore = Depends(get_store),
) -> ReferenceStatsResponse:
    catalog = await run_in_threadpool(store.catalog, user.id)
    counts = Counter(r.flavor_id for r in catalog.references)
    return ReferenceStatsResponse(
        flavors=[
            FlavorStats(flavor_id=f.id, slug=f.slug, name=f.name, references=counts.get(f.id, 0))
            for f in catalog.flavors
        ]
    )


@router.delete("/{reference_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_reference(
    reference_id: str,
    user: CurrentUser = Depends(get_current_user),
    store: ReferenceStore = Depends(get_store),
) -> None:
    if not await run_in_threadpool(store.delete_reference, user.id, reference_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Referencia no encontrada")
