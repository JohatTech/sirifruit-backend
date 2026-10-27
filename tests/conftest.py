"""Shared test fixtures: synthetic pack images, fake embedder, in-memory store, auth tokens."""
import time

import cv2
import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.ml.matcher import FlavorInfo
from app.services.reference_store import InMemoryReferenceStore

SECRET = "test-secret-with-at-least-32-characters!!"
USER_ID = "11111111-1111-1111-1111-111111111111"

# BGR colors that imitate each flavor's pack
FLAVOR_COLORS = {
    "fresa": (60, 40, 220),
    "pina": (40, 210, 240),
    "maracuya": (30, 140, 245),
    "tamarindo": (40, 80, 120),
    "coco": (225, 235, 240),
    "choco_galleta": (30, 40, 60),
}
FLAVORS = [
    FlavorInfo(f"id-{slug}", slug, slug.replace("_", " ").title()) for slug in FLAVOR_COLORS
]


def make_pack(slug: str, seed: int = 0, size: int = 480) -> bytes:
    """A colored 'pack' on a light background with slight random offset, rotation and lighting."""
    rng = np.random.default_rng(seed)
    bg = np.full((size, size, 3), 235, np.uint8)
    color = np.array(FLAVOR_COLORS[slug], np.float32) * rng.uniform(0.85, 1.1)
    w, h = int(size * rng.uniform(0.35, 0.5)), int(size * rng.uniform(0.5, 0.65))
    cx = size // 2 + int(rng.integers(-30, 30))
    cy = size // 2 + int(rng.integers(-30, 30))
    box = cv2.boxPoints(((cx, cy), (w, h), float(rng.uniform(-15, 15)))).astype(np.int32)
    cv2.fillPoly(bg, [box], tuple(int(c) for c in np.clip(color, 0, 255)))
    # A "label" stripe so packs have structure, not only color.
    cv2.line(bg, tuple(box[0]), tuple(box[2]), (250, 250, 250), 6)
    noise = rng.normal(0, 4, bg.shape)
    img = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


class FakeEmbedder:
    """Deterministic cheap embedding: downsampled image pixels (no torch needed)."""

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(rgb, (12, 16), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
        v = small - small.mean()
        return (v / (np.linalg.norm(v) + 1e-6)).astype(np.float32)


def make_token(sub: str = USER_ID, secret: str = SECRET, exp_offset: int = 3600) -> str:
    return jwt.encode(
        {"sub": sub, "aud": "authenticated", "exp": int(time.time()) + exp_offset, "role": "authenticated"},
        secret,
        algorithm="HS256",
    )


@pytest.fixture
def store() -> InMemoryReferenceStore:
    s = InMemoryReferenceStore()
    s.set_flavors(USER_ID, FLAVORS)
    return s


@pytest.fixture
def client(store):
    app = create_app()
    app.state.embedder = FakeEmbedder()
    app.state.store = store
    app.dependency_overrides[get_settings] = lambda: Settings(supabase_jwt_secret=SECRET)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth() -> dict:
    return {"Authorization": f"Bearer {make_token()}"}
