import sys
from pathlib import Path
from datasets import load_from_disk

# Configuration
RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"


def safe_str(val):
    return str(val) if val is not None else ""


def preprocess_codealpaca(example):
    user_msg = safe_str(example.get("instruction", ""))
    inp = safe_str(example.get("input", ""))
    if inp.strip():
        user_msg += "\n\n" + inp
    
    return {
        "messages": [
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": safe_str(example.get("output", "")).strip()}
        ]
    }


def preprocess_opencoder(example):
    return {
        "messages": [
            {"role": "user", "content": safe_str(example.get("instruction", "")).strip()},
            {"role": "assistant", "content": safe_str(example.get("output", "")).strip()}
        ]
    }


def preprocess_classeval(example):
    desc = safe_str(example.get("class_description", ""))
    skel = safe_str(example.get("skeleton", ""))
    user_msg = f"{desc}\n\nComplete the following Python code:\n{skel}"
    
    return {
        "messages": [
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": safe_str(example.get("solution_code", "")).strip()}
        ]
    }


def is_valid(example):
    messages = example.get("messages", [])
    if len(messages) != 2:
        return False
    if not messages[0]["content"] or not messages[1]["content"]:
        return False
    return True


def process_dataset(dataset_name):
    raw_path = RAW_DIR / dataset_name
    processed_path = PROCESSED_DIR / f"{dataset_name}_chat"
    
    # Skip if folder doesn't exist or doesn't look like a real dataset folder
    if not raw_path.exists() or not raw_path.is_dir() or dataset_path_is_empty(raw_path):
        print(f"[SKIP] {dataset_name} (not found or empty)")
        return
        
    try:
        ds = load_from_disk(str(raw_path))
    except Exception as e:
        print(f"[FAIL] {dataset_name}: failed to load -- {e}")
        return

    print(f"\n[PROC] Processing {dataset_name} (Examples: {len(ds)})...")

    # Determine mapping function based on name
    if "codealpaca" in dataset_name:
        map_fn = preprocess_codealpaca
    elif "opencoder" in dataset_name:
        map_fn = preprocess_opencoder
    elif "classeval" in dataset_name:
        map_fn = preprocess_classeval
    else:
        print(f"  [WARN] No mapping function defined for {dataset_name}. Skipping.")
        return

    # Map to ChatML format, dropping all original columns
    ds_chat = ds.map(
        map_fn,
        remove_columns=ds.column_names,
        desc=f"Formatting {dataset_name}"
    )

    # Filter out empty examples
    original_len = len(ds_chat)
    ds_chat = ds_chat.filter(is_valid, desc=f"Filtering {dataset_name}")
    filtered_len = len(ds_chat)

    print(f"  [INFO] Kept {filtered_len} / {original_len} examples.")

    # Save processed dataset
    processed_path.mkdir(parents=True, exist_ok=True)
    ds_chat.save_to_disk(str(processed_path))
    print(f"  [OK] Saved to {processed_path}")


def dataset_path_is_empty(path: Path) -> bool:
    """Helper to check if a directory only contains a .gitkeep or is entirely empty."""
    files = [f for f in path.iterdir() if f.name != ".gitkeep"]
    return len(files) == 0


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    targets = sys.argv[1:]
    
    print("=" * 60)
    print("  Dataset Preprocessing")
    print("=" * 60)
    
    if targets:
        for t in targets:
            process_dataset(t)
    else:
        # Process all available in raw
        for p in sorted(RAW_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                process_dataset(p.name)

    print("\n" + "=" * 60)
    print("  Preprocessing complete.")


if __name__ == "__main__":
    main()
