"""Download all coding datasets from HuggingFace Hub into datasets/raw/.

Usage:
    python scripts/download_datasets.py              # download all
    python scripts/download_datasets.py codealpaca   # download one
"""

import sys
import time
from datasets import load_dataset
from pathlib import Path
import os
from huggingface_hub import login

# ── Configuration ──────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"

DATASETS = {
    "opencoder_stage1": {
        "hub_id": "OpenCoder-LLM/opc-sft-stage1",
        "subset": ["realuser_instruct", "filtered_infinity_instruct", "largescale_diverse_instruct"],
        "split": "train",
        "description": "OpenCoder SFT Stage 1 — foundational coding pairs",
    },
    "opencoder_stage2": {
        "hub_id": "OpenCoder-LLM/opc-sft-stage2",
        "subset": ["educational_instruct", "evol_instruct", "mceval_instruct", "package_instruct"],
        "split": "train",
        "description": "OpenCoder SFT Stage 2 — advanced coding pairs",
    },
    "codealpaca": {
        "hub_id": "sahil2801/CodeAlpaca-20k",
        "subset": None,
        "split": "train",
        "description": "20k instruction-following coding examples",
    },
    "apps": {
        "hub_id": "codeparrot/apps",
        "subset": None,
        "split": "train",
        "description": "10k competitive programming problems (filter before training)",
    },
    "codesearchnet": {
        "hub_id": "code_search_net",
        "subset": "python",
        "split": "train",
        "description": "Code-docstring pairs (python subset) — code, func_documentation_string, language",
    },
    "classeval": {
        "hub_id": "FudanSELab/ClassEval",
        "subset": None,
        "split": "test",
        "eval_only": True,
        "description": "Class-level code generation benchmark (EVAL ONLY — 100 tasks, no train split)",
    },
}
# ───────────────────────────────────────────────────────────────


def check_auth():
    """Check Hugging Face authentication for protected datasets."""
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            return True
        except Exception as e:
            print(f"  [WARN] HF auth failed: {e}")
            return False
    return False


def download_one(name, info):
    """Download a single dataset and save to disk."""
    save_path = RAW_DIR / name
    if save_path.exists() and any(save_path.iterdir()):
        # Skip if already has data (beyond .gitkeep)
        real_files = [f for f in save_path.iterdir() if f.name != ".gitkeep"]
        if real_files:
            print(f"  [SKIP] {name}: already downloaded, skipping (delete folder to re-download)")
            return

    save_path.mkdir(parents=True, exist_ok=True)

    hub_id = info["hub_id"]
    subset = info.get("subset")
    split = info.get("split", "train")

    print(f"  [DOWN] {name}: downloading {hub_id} (split={split})...")
    if info.get("eval_only"):
        print(f"      [WARN] EVAL ONLY -- do not include in training data")
    start = time.time()

    try:
        from datasets import concatenate_datasets
        if isinstance(subset, list):
            datasets = []
            for sub in subset:
                print(f"      [DOWN] downloading subset: {sub}...")
                ds = load_dataset(hub_id, sub, split=split)
                datasets.append(ds)
            ds = concatenate_datasets(datasets)
        else:
            ds = load_dataset(hub_id, subset, split=split)
            
        ds.save_to_disk(str(save_path))
        elapsed = time.time() - start
        print(f"  [OK] {name}: {len(ds)} examples saved ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  [FAIL] {name}: download failed -- {e}")


def main():
    # Check authentication
    check_auth()
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Allow downloading a specific dataset by name
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    print("=" * 60)
    print("  Dataset Download Pipeline")
    print("=" * 60)
    print(f"  Target directory: {RAW_DIR}")
    print(f"  Datasets: {', '.join(targets)}")
    print("=" * 60)

    for name in targets:
        if name not in DATASETS:
            print(f"  [WARN] Unknown dataset: {name}")
            print(f"     Available: {', '.join(DATASETS.keys())}")
            continue
        download_one(name, DATASETS[name])

    print("\n  Done.")


if __name__ == "__main__":
    main()