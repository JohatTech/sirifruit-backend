"""Few-shot matcher: compares a query against the flavor reference database."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlavorInfo:
    id: str
    slug: str
    name: str


@dataclass(frozen=True)
class Reference:
    id: str
    flavor_id: str
    embedding: np.ndarray  # L2-normalized
    hist: np.ndarray       # L1-normalized


@dataclass(frozen=True)
class Prediction:
    flavor_id: str
    slug: str
    name: str
    confidence: float
    score: float


@dataclass(frozen=True)
class MatchResult:
    predictions: list[Prediction]
    low_confidence: bool
    references_used: int


def _hist_similarity(query: np.ndarray, hists: np.ndarray) -> np.ndarray:
    """Bhattacharyya coefficient in [0, 1] between L1-normalized histograms."""
    return np.sqrt(hists * query[None, :]).sum(axis=1)


def match(
    query_emb: np.ndarray,
    query_hist: np.ndarray,
    references: list[Reference],
    flavors: list[FlavorInfo],
    *,
    embedding_weight: float = 0.7,
    top_k: int = 3,
    temperature: float = 0.05,
    conf_threshold: float = 0.6,
    margin_threshold: float = 0.25,
    max_results: int = 3,
) -> MatchResult:
    flavor_by_id = {f.id: f for f in flavors}
    refs = [r for r in references if r.flavor_id in flavor_by_id]
    if not refs:
        return MatchResult(predictions=[], low_confidence=True, references_used=0)

    embs = np.stack([r.embedding for r in refs])
    hists = np.stack([r.hist for r in refs])
    cos = embs @ query_emb
    col = _hist_similarity(query_hist, hists)
    scores = embedding_weight * cos + (1.0 - embedding_weight) * col

    # Aggregate per flavor: mean of the best k reference scores.
    per_flavor: dict[str, float] = {}
    for fid in {r.flavor_id for r in refs}:
        idx = [i for i, r in enumerate(refs) if r.flavor_id == fid]
        best = np.sort(scores[idx])[::-1][:top_k]
        per_flavor[fid] = float(best.mean())

    ids = list(per_flavor)
    vals = np.array([per_flavor[i] for i in ids], dtype=np.float64)
    exp = np.exp((vals - vals.max()) / max(temperature, 1e-6))
    probs = exp / exp.sum()
    order = np.argsort(probs)[::-1]

    preds = [
        Prediction(
            flavor_id=ids[i],
            slug=flavor_by_id[ids[i]].slug,
            name=flavor_by_id[ids[i]].name,
            confidence=round(float(probs[i]), 4),
            score=round(float(vals[i]), 4),
        )
        for i in order[:max_results]
    ]
    top = preds[0].confidence
    second = preds[1].confidence if len(preds) > 1 else 0.0
    low = top < conf_threshold or (top - second) < margin_threshold
    return MatchResult(predictions=preds, low_confidence=low, references_used=len(refs))
