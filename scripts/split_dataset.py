"""Split datasets into train / validation / test sets."""

from datasets import load_from_disk
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
INPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"

TRAIN_DIR = Path(__file__).resolve().parent.parent / "datasets" / "train"
VAL_DIR = Path(__file__).resolve().parent.parent / "datasets" / "validation"
TEST_DIR = Path(__file__).resolve().parent.parent / "datasets" / "test"

VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42
# ───────────────────────────────────────────────────────────────


def split_dataset(dataset_path):
    """Split a single dataset into train/val/test."""
    ds = load_from_disk(str(dataset_path))
    name = dataset_path.name

    # First split: separate test set
    split1 = ds.train_test_split(test_size=TEST_RATIO, seed=SEED)
    test_ds = split1["test"]

    # Second split: separate validation from remaining train
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
    split2 = split1["train"].train_test_split(test_size=val_ratio_adjusted, seed=SEED)
    train_ds = split2["train"]
    val_ds = split2["test"]

    print(f"  {name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    for dataset_path in sorted(INPUT_DIR.iterdir()):
        if dataset_path.is_dir() and dataset_path.name != ".gitkeep":
            train_ds, val_ds, test_ds = split_dataset(dataset_path)

            train_ds.save_to_disk(str(TRAIN_DIR / dataset_path.name))
            val_ds.save_to_disk(str(VAL_DIR / dataset_path.name))
            test_ds.save_to_disk(str(TEST_DIR / dataset_path.name))

    print("Splitting complete.")
