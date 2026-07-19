"""Tokenize ChatML datasets using the model tokenizer."""

from datasets import load_from_disk
from transformers import AutoTokenizer
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

TRAIN_DIR = Path(__file__).resolve().parent.parent / "datasets" / "train"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "train"

MAX_SEQ_LENGTH = 2048
# ───────────────────────────────────────────────────────────────


def tokenize_examples(examples, tokenizer):
    """Apply chat template and tokenize a batch of examples."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


if __name__ == "__main__":
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_path in sorted(TRAIN_DIR.iterdir()):
        if dataset_path.is_dir() and dataset_path.name != ".gitkeep":
            ds = load_from_disk(str(dataset_path))

            ds = ds.map(
                lambda examples: tokenize_examples(examples, tokenizer),
                batched=True,
                remove_columns=ds.column_names,
            )

            save_path = OUTPUT_DIR / f"{dataset_path.name}_tokenized"
            ds.save_to_disk(str(save_path))
            print(f"  {dataset_path.name}: {len(ds)} examples tokenized (max_len={MAX_SEQ_LENGTH})")

    print("Tokenization complete.")
