import sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

# Configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"
TOKENIZED_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tokenized"

MAX_SEQ_LENGTH = 2048


def tokenize_examples(examples, tokenizer):
    """Apply chat template and tokenize a batch of examples."""
    texts = []
    for messages in examples["messages"]:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        except Exception as e:
            fallback_text = ""
            for msg in messages:
                fallback_text += f"{msg['role']}: {msg['content']}\n"
            texts.append(fallback_text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def process_dataset(dataset_name, tokenizer):
    dataset_path = PROCESSED_DIR / dataset_name
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"  [SKIP] {dataset_name} not found in {PROCESSED_DIR}")
        return

    print(f"\n[PROC] Loading {dataset_name}...")
    try:
        ds = load_from_disk(str(dataset_path))
    except Exception as e:
        print(f"  [FAIL] Failed to load {dataset_name}: {e}")
        return

    print(f"       Tokenizing (max_length={MAX_SEQ_LENGTH})...")
    ds_tokenized = ds.map(
        lambda examples: tokenize_examples(examples, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {dataset_name}"
    )

    out_name = dataset_name.replace("_chat", "_tokenized") if "_chat" in dataset_name else f"{dataset_name}_tokenized"
    save_path = TOKENIZED_DIR / out_name
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    
    ds_tokenized.save_to_disk(str(save_path))
    print(f"  [OK] Saved tokenized dataset to {save_path}")


def main():
    print("=" * 60)
    print("  Dataset Tokenization")
    print("=" * 60)
    
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False, cache_dir="./.hf_cache")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    targets = sys.argv[1:]
    
    if targets:
        for t in targets:
            process_dataset(t, tokenizer)
    else:
        if not PROCESSED_DIR.exists():
            print(f"[FAIL] Processed directory not found: {PROCESSED_DIR}")
            return
            
        for p in sorted(PROCESSED_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                process_dataset(p.name, tokenizer)
                
    print("\n" + "=" * 60)
    print("  Tokenization complete.")


if __name__ == "__main__":
    main()
