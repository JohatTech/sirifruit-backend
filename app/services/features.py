import numpy as np

from app.ml.embedder import Embedder
from app.ml.preprocess import preprocess


def extract_features(data: bytes, embedder: Embedder) -> tuple[np.ndarray, np.ndarray]:
    """Bytes -> (CNN embedding, HSV histogram). Same pipeline for enrollment and prediction."""
    rgb, hist = preprocess(data)
    return embedder.embed(rgb), hist
