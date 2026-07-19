"""Convert instruction/output format to ChatML messages format."""

from datasets import load_from_disk
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
INPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"

INSTRUCTION_COL = "instruction"
INPUT_COL = "input"
RESPONSE_COL = "output"

SYSTEM_PROMPT = (
    "You are an expert coding assistant. Provide clear, correct, "
    "and well-documented code with explanations."
)
# ───────────────────────────────────────────────────────────────


def to_chatml(example):
    """Convert a single example to ChatML messages format."""
    instruction = example[INSTRUCTION_COL]

    # Append optional input context to the instruction
    extra_input = example.get(INPUT_COL, "")
    if extra_input and extra_input.strip():
        instruction = f"{instruction}\n\n{extra_input.strip()}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": example[RESPONSE_COL]},
    ]

    return {"messages": messages}


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_path in sorted(INPUT_DIR.iterdir()):
        if dataset_path.is_dir() and dataset_path.name != ".gitkeep":
            ds = load_from_disk(str(dataset_path))

            # Remove old columns, keep only messages
            ds = ds.map(to_chatml, remove_columns=ds.column_names)

            save_path = OUTPUT_DIR / dataset_path.name
            ds.save_to_disk(str(save_path))
            print(f"  {dataset_path.name}: {len(ds)} examples -> ChatML")

    print("Conversion complete.")
