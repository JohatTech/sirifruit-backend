"""Bulk-load reference photos into Supabase for one user.

Folder layout:  <root>/<slug>/*.jpg   (slug = coco, fresa, pina, maracuya, tamarindo, choco_galleta)

    python -m scripts.enroll_folder references/ --user-id <uuid-del-usuario>
"""
import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.ml.embedder import MobileNetEmbedder
from app.services.features import extract_features
from app.services.reference_store import SupabaseReferenceStore

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--user-id", required=True)
    args = ap.parse_args()

    s = get_settings()
    store = SupabaseReferenceStore(s.supabase_url, s.supabase_service_key, s.storage_bucket)
    by_slug = {f.slug: f for f in store.catalog(args.user_id).flavors}
    if not by_slug:
        print("El usuario no tiene sabores. ¿Ejecutaste la migración y seed.sql?")
        return 1

    embedder = MobileNetEmbedder(threads=s.torch_threads)
    total = 0
    for d in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        flavor = by_slug.get(d.name)
        if flavor is None:
            print(f"! Carpeta '{d.name}' no coincide con ningún sabor, se omite")
            continue
        files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in IMAGE_EXT]
        for f in files:
            data = f.read_bytes()
            emb, hist = extract_features(data, embedder)
            store.add_reference(args.user_id, flavor.id, emb, hist, data, "enroll")
        total += len(files)
        print(f"✓ {flavor.name}: {len(files)} fotos")
    print(f"Listo: {total} referencias cargadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
