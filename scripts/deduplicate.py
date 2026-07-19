"""Deduplicate dataset examples by instruction hash."""

import hashlib
from datasets import load_from_disk
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
INPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"

DEDUP_COLUMN = "instruction"
# ───────────────────────────────────────────────────────────────


def hash_text(text):
    """Return SHA-256 hex digest of a string."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def deduplicate(dataset_path):
    """Remove duplicate examples from a dataset."""
    ds = load_from_disk(str(dataset_path))
    original_len = len(ds)

    seen = set()
    keep_indices = []

    for i, example in enumerate(ds):
        h = hash_text(example[DEDUP_COLUMN])
        if h not in seen:
            seen.add(h)
            keep_indices.append(i)

    ds = ds.select(keep_indices)
    print(f"  {dataset_path.name}: {original_len} -> {len(ds)} examples (removed {original_len - len(ds)} duplicates)")
    return ds


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_path in sorted(INPUT_DIR.iterdir()):
        if dataset_path.is_dir() and dataset_path.name != ".gitkeep":
            ds = deduplicate(dataset_path)
            ds.save_to_disk(str(OUTPUT_DIR / dataset_path.name))

    print("Deduplication complete.")
