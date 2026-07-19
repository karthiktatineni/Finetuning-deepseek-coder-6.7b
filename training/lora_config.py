"""LoRA adapter configuration for PEFT fine-tuning."""

from peft import LoraConfig, TaskType


def get_lora_config():
    """Return the LoRA configuration for Qwen2.5-Coder."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
