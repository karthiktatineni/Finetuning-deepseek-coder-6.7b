"""Generate aggregate statistics across all datasets.

Usage:
    python scripts/stats.py              # stats for datasets/raw/
    python scripts/stats.py processed    # stats for datasets/processed/
"""

import sys
from datasets import load_from_disk
from pathlib import Path
from collections import Counter

# ── Configuration ──────────────────────────────────────────────
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
# ───────────────────────────────────────────────────────────────


def compute_stats(dataset_path):
    """Compute stats for a single dataset."""
    try:
        ds = load_from_disk(str(dataset_path))
    except Exception:
        return None

    stats = {
        "name": dataset_path.name,
        "num_examples": len(ds),
        "columns": ds.column_names,
    }

    # Compute text column stats
    text_stats = {}
    for col in ds.column_names:
        sample = ds[0][col]
        if isinstance(sample, str):
            sample_size = min(1000, len(ds))
            lengths = [len(str(ds[i][col])) for i in range(sample_size)]
            empty = sum(1 for l in lengths if l == 0)
            text_stats[col] = {
                "avg_length": sum(lengths) / len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "empty_count": empty,
                "empty_pct": round(empty / len(lengths) * 100, 1),
            }

    stats["text_columns"] = text_stats
    return stats


def print_summary_table(all_stats):
    """Print a formatted summary table."""
    print(f"\n{'Dataset':<25s} {'Examples':>10s} {'Columns':>10s}")
    print("-" * 50)

    total = 0
    for s in all_stats:
        print(f"{s['name']:<25s} {s['num_examples']:>10,d} {len(s['columns']):>10d}")
        total += s["num_examples"]

    print("-" * 50)
    print(f"{'TOTAL':<25s} {total:>10,d}")


def print_text_details(all_stats):
    """Print text column statistics per dataset."""
    print(f"\n{'Dataset':<20s} {'Column':<25s} {'Avg Len':>8s} {'Min':>6s} {'Max':>8s} {'Empty%':>7s}")
    print("-" * 80)

    for s in all_stats:
        for col, ts in s.get("text_columns", {}).items():
            print(
                f"{s['name']:<20s} {col:<25s} "
                f"{ts['avg_length']:>8.0f} {ts['min_length']:>6d} "
                f"{ts['max_length']:>8d} {ts['empty_pct']:>6.1f}%"
            )


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "raw"
    target_dir = DATASETS_DIR / stage

    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return

    print("=" * 60)
    print(f"  Dataset Statistics -- {stage}/")
    print("=" * 60)

    all_stats = []
    for dataset_path in sorted(target_dir.iterdir()):
        if not dataset_path.is_dir() or dataset_path.name == ".gitkeep":
            continue

        real_files = [f for f in dataset_path.iterdir() if f.name != ".gitkeep"]
        if not real_files:
            continue

        stats = compute_stats(dataset_path)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print(f"\n  No datasets found in {target_dir}")
        return

    print_summary_table(all_stats)
    print_text_details(all_stats)

    print(f"\n{'=' * 60}")
    print("  Stats complete.")


if __name__ == "__main__":
    main()
