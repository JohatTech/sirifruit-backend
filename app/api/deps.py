from fastapi import Depends, File, HTTPException, Request, UploadFile, status

from app.core.config import Settings, get_settings
from app.ml.embedder import Embedder
from app.services.reference_store import ReferenceStore

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "application/octet-stream"}


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_store(request: Request) -> ReferenceStore:
    return request.app.state.store


async def read_image(file: UploadFile, settings: Settings) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Solo se aceptan imágenes JPEG o PNG")
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "La imagen supera el tamaño máximo")
    return data


async def image_upload(
    file: UploadFile = File(...), settings: Settings = Depends(get_settings)
) -> bytes:
    return await read_image(file, settings)
