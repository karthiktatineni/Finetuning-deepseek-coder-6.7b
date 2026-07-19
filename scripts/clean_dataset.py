"""Clean and filter raw datasets."""

from datasets import load_from_disk
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
INPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"

MIN_INSTRUCTION_LEN = 10
MIN_RESPONSE_LEN = 20

INSTRUCTION_COL = "instruction"
RESPONSE_COL = "output"
# ───────────────────────────────────────────────────────────────


def clean_example(example):
    """Strip whitespace from text fields."""
    example[INSTRUCTION_COL] = example[INSTRUCTION_COL].strip()
    example[RESPONSE_COL] = example[RESPONSE_COL].strip()
    return example


def is_valid(example):
    """Filter out empty or too-short examples."""
    instruction = example.get(INSTRUCTION_COL, "")
    response = example.get(RESPONSE_COL, "")
    return len(instruction) >= MIN_INSTRUCTION_LEN and len(response) >= MIN_RESPONSE_LEN


def clean_dataset(dataset_path):
    """Load, clean, and filter a single dataset."""
    ds = load_from_disk(str(dataset_path))
    name = dataset_path.name

    original_len = len(ds)
    ds = ds.map(clean_example)
    ds = ds.filter(is_valid)

    print(f"  {name}: {original_len} -> {len(ds)} examples")
    return ds


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_path in sorted(INPUT_DIR.iterdir()):
        if dataset_path.is_dir() and dataset_path.name != ".gitkeep":
            ds = clean_dataset(dataset_path)
            ds.save_to_disk(str(OUTPUT_DIR / dataset_path.name))

    print("Cleaning complete.")
