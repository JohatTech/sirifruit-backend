"""Offline evaluation of the recognition pipeline (leave-one-out over a photo folder).

Folder layout:  <root>/<slug>/*.jpg   e.g. references/fresa/001.jpg

    python -m scripts.evaluate references/
    python -m scripts.evaluate --synthetic        # sanity check with generated packs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.ml.matcher import FlavorInfo, Reference, match
from app.services.features import extract_features

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def load_folder(root: Path) -> list[tuple[str, bytes]]:
    items = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                items.append((d.name, f.read_bytes()))
    return items


def load_synthetic(per_flavor: int) -> list[tuple[str, bytes]]:
    from tests.conftest import FLAVOR_COLORS, make_pack

    return [(slug, make_pack(slug, seed=i)) for slug in FLAVOR_COLORS for i in range(per_flavor)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--per-flavor", type=int, default=8)
    ap.add_argument("--fake-embedder", action="store_true", help="skip torch (debug only)")
    args = ap.parse_args()

    if args.synthetic:
        items = load_synthetic(args.per_flavor)
    elif args.folder:
        items = load_folder(args.folder)
    else:
        ap.error("indica una carpeta o --synthetic")
    if not items:
        print("No se encontraron imágenes")
        return 1

    if args.fake_embedder:
        from tests.conftest import FakeEmbedder

        embedder = FakeEmbedder()
    else:
        from app.ml.embedder import MobileNetEmbedder

        embedder = MobileNetEmbedder()

    s = get_settings()
    slugs = sorted({slug for slug, _ in items})
    flavors = [FlavorInfo(slug, slug, slug) for slug in slugs]

    t0 = time.perf_counter()
    feats = [(slug, *extract_features(data, embedder)) for slug, data in items]
    per_img_ms = (time.perf_counter() - t0) / len(items) * 1000

    refs = [Reference(str(i), slug, e, h) for i, (slug, e, h) in enumerate(feats)]
    idx = {s_: i for i, s_ in enumerate(slugs)}
    confusion = np.zeros((len(slugs), len(slugs)), int)
    low = 0
    for i, (slug, e, h) in enumerate(feats):
        others = [r for r in refs if r.id != str(i)]
        res = match(
            e, h, others, flavors,
            embedding_weight=s.embedding_weight, top_k=s.top_k_per_flavor,
            temperature=s.softmax_temperature, conf_threshold=s.conf_threshold,
            margin_threshold=s.margin_threshold,
        )
        confusion[idx[slug], idx[res.predictions[0].slug]] += 1
        low += res.low_confidence

    acc = np.trace(confusion) / confusion.sum()
    print(f"Imágenes: {len(items)}  ·  Sabores: {len(slugs)}  ·  {per_img_ms:.0f} ms/imagen")
    print(f"Precisión top-1 (leave-one-out): {acc:.1%}  ·  Baja confianza: {low}/{len(items)}\n")
    width = max(len(x) for x in slugs) + 2
    print(" " * width + "".join(f"{x[:8]:>9}" for x in slugs))
    for i, slug in enumerate(slugs):
        print(f"{slug:<{width}}" + "".join(f"{v:>9}" for v in confusion[i]))
    return 0 if acc >= 0.9 else 2


if __name__ == "__main__":
    sys.exit(main())
