"""Training configuration for Qwen2.5-Coder fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainingConfig:
    """Central training configuration."""

    # ── Model ──────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # ── Data ───────────────────────────────────────────────────
    train_data: str = str(PROJECT_ROOT / "datasets" / "train")
    val_data: str = str(PROJECT_ROOT / "datasets" / "validation")
    max_seq_length: int = 2048

    # ── Training hyperparameters ───────────────────────────────
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # ── Precision & optimization ───────────────────────────────
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # ── Logging & saving ───────────────────────────────────────
    output_dir: str = str(PROJECT_ROOT / "checkpoints")
    logging_dir: str = str(PROJECT_ROOT / "logs")
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 3

    # ── Misc ───────────────────────────────────────────────────
    seed: int = 42
    report_to: str = "none"
