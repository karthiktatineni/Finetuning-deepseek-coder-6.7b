"""Inspect raw datasets — structure, columns, sizes, samples.

Usage:
    python scripts/inspect_dataset.py                # inspect all
    python scripts/inspect_dataset.py codealpaca     # inspect one
"""

import sys
import json
from datasets import load_from_disk
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
# ───────────────────────────────────────────────────────────────


def inspect_one(dataset_path):
    """Print a detailed summary of a single dataset."""
    name = dataset_path.name

    try:
        ds = load_from_disk(str(dataset_path))
    except Exception as e:
        print(f"\n  [FAIL] {name}: failed to load -- {e}")
        return

    print(f"\n{'-' * 60}")
    print(f"  [DS] {name}")
    print(f"{'-' * 60}")
    print(f"  Examples:  {len(ds)}")
    print(f"  Columns:   {ds.column_names}")
    print(f"  Features:  {ds.features}")

    # Column-level stats
    print(f"\n  Column details:")
    for col in ds.column_names:
        sample_val = ds[0][col]
        dtype = type(sample_val).__name__

        if isinstance(sample_val, str):
            lengths = [len(str(ds[i][col])) for i in range(min(500, len(ds)))]
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            empty = sum(1 for l in lengths if l == 0)
            print(f"    {col:25s}  type={dtype:8s}  avg_len={avg_len:.0f}  min={min_len}  max={max_len}  empty={empty}")
        elif isinstance(sample_val, list):
            avg_items = sum(len(ds[i][col]) for i in range(min(500, len(ds)))) / min(500, len(ds))
            print(f"    {col:25s}  type=list      avg_items={avg_items:.1f}")
        else:
            print(f"    {col:25s}  type={dtype}")

    # Show first example
    print(f"\n  First example:")
    first = ds[0]
    for k, v in first.items():
        display = str(v)
        if len(display) > 200:
            display = display[:200] + "..."
        print(f"    {k}: {display}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  Dataset Inspector")
    print("=" * 60)

    for dataset_path in sorted(RAW_DIR.iterdir()):
        if not dataset_path.is_dir() or dataset_path.name == ".gitkeep":
            continue
        if targets and dataset_path.name not in targets:
            continue

        # Check if it has actual data (not just .gitkeep)
        real_files = [f for f in dataset_path.iterdir() if f.name != ".gitkeep"]
        if not real_files:
            print(f"\n  [SKIP] {dataset_path.name}: empty (not yet downloaded)")
            continue

        inspect_one(dataset_path)

    if targets:
        for t in targets:
            target_path = Path(t)
            if "/" in t or "\\" in t or (target_path.parent.name == "processed"):
                # It's a path (e.g. processed/opencoder_stage2_chat)
                p = Path(__file__).resolve().parent.parent / "datasets" / t
                if p.exists() and p.is_dir():
                    inspect_one(p)

    print(f"\n{'=' * 60}")
    print("  Inspection complete.")


if __name__ == "__main__":
    main()
