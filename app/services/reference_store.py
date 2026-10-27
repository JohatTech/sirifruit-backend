"""Persistence of the recognition database (Supabase) with a per-user in-memory cache."""
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.ml.matcher import FlavorInfo, Reference


@dataclass
class UserCatalog:
    flavors: list[FlavorInfo] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


class ReferenceStore(Protocol):
    def catalog(self, user_id: str) -> UserCatalog: ...

    def add_reference(
        self,
        user_id: str,
        flavor_id: str,
        embedding: np.ndarray,
        hist: np.ndarray,
        image: bytes | None,
        source: str,
    ) -> str: ...

    def delete_reference(self, user_id: str, reference_id: str) -> bool: ...


class InMemoryReferenceStore:
    """Used by tests and offline evaluation."""

    def __init__(self) -> None:
        self._data: dict[str, UserCatalog] = {}

    def set_flavors(self, user_id: str, flavors: list[FlavorInfo]) -> None:
        self._data.setdefault(user_id, UserCatalog()).flavors = list(flavors)

    def catalog(self, user_id: str) -> UserCatalog:
        return self._data.setdefault(user_id, UserCatalog())

    def add_reference(self, user_id, flavor_id, embedding, hist, image, source) -> str:
        rid = str(uuid.uuid4())
        self.catalog(user_id).references.append(Reference(rid, flavor_id, embedding, hist))
        return rid

    def delete_reference(self, user_id, reference_id) -> bool:
        cat = self.catalog(user_id)
        before = len(cat.references)
        cat.references = [r for r in cat.references if r.id != reference_id]
        return len(cat.references) < before


def _parse_vector(value) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


class SupabaseReferenceStore:
    PAGE = 1000

    def __init__(self, url: str, service_key: str, bucket: str) -> None:
        from supabase import create_client

        self._client = create_client(url, service_key)
        self._bucket = bucket
        self._cache: dict[str, UserCatalog] = {}
        self._lock = threading.Lock()

    def invalidate(self, user_id: str) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def _load(self, user_id: str) -> UserCatalog:
        rows = (
            self._client.table("flavors")
            .select("id,slug,display_name")
            .eq("owner_id", user_id)
            .eq("active", True)
            .execute()
            .data
        )
        flavors = [FlavorInfo(r["id"], r["slug"], r["display_name"]) for r in rows]

        refs: list[Reference] = []
        start = 0
        while True:
            page = (
                self._client.table("flavor_references")
                .select("id,flavor_id,embedding,hsv_hist")
                .eq("owner_id", user_id)
                .order("created_at")
                .range(start, start + self.PAGE - 1)
                .execute()
                .data
            )
            refs += [
                Reference(
                    r["id"],
                    r["flavor_id"],
                    _parse_vector(r["embedding"]),
                    np.asarray(r["hsv_hist"], dtype=np.float32),
                )
                for r in page
            ]
            if len(page) < self.PAGE:
                break
            start += self.PAGE
        return UserCatalog(flavors, refs)

    def catalog(self, user_id: str) -> UserCatalog:
        with self._lock:
            cached = self._cache.get(user_id)
        if cached is not None:
            return cached
        cat = self._load(user_id)
        with self._lock:
            self._cache[user_id] = cat
        return cat

    def add_reference(self, user_id, flavor_id, embedding, hist, image, source) -> str:
        rid = str(uuid.uuid4())
        image_path = None
        if image:
            slug = next((f.slug for f in self.catalog(user_id).flavors if f.id == flavor_id), "otro")
            image_path = f"{user_id}/refs/{slug}/{rid}.jpg"
            self._client.storage.from_(self._bucket).upload(
                image_path, image, {"content-type": "image/jpeg"}
            )
        self._client.table("flavor_references").insert(
            {
                "id": rid,
                "owner_id": user_id,
                "flavor_id": flavor_id,
                "embedding": "[" + ",".join(f"{v:.6f}" for v in embedding.tolist()) + "]",
                "hsv_hist": [round(float(v), 6) for v in hist.tolist()],
                "image_path": image_path,
                "source": source,
            }
        ).execute()
        self.invalidate(user_id)
        return rid

    def delete_reference(self, user_id, reference_id) -> bool:
        res = (
            self._client.table("flavor_references")
            .delete()
            .eq("id", reference_id)
            .eq("owner_id", user_id)
            .execute()
        )
        self.invalidate(user_id)
        return bool(res.data)
