"""Custom trainer setup for LoRA fine-tuning."""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

from config import TrainingConfig
from lora_config import get_lora_config


def load_quantized_model(config: TrainingConfig):
    """Load model in 4-bit quantization with LoRA adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    return model


def load_tokenizer(config: TrainingConfig):
    """Load tokenizer and set pad token."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_datasets(config: TrainingConfig):
    """Load train and validation datasets from disk."""
    train_ds = None
    val_ds = None

    train_path = config.train_data
    val_path = config.val_data

    # Load first available dataset from train/val directories
    from pathlib import Path
    for p in sorted(Path(train_path).iterdir()):
        if p.is_dir() and p.name != ".gitkeep":
            train_ds = load_from_disk(str(p))
            print(f"Train: {p.name} ({len(train_ds)} examples)")
            break

    for p in sorted(Path(val_path).iterdir()):
        if p.is_dir() and p.name != ".gitkeep":
            val_ds = load_from_disk(str(p))
            print(f"Val:   {p.name} ({len(val_ds)} examples)")
            break

    return train_ds, val_ds


def create_trainer(model, tokenizer, train_ds, val_ds, config: TrainingConfig):
    """Build a HuggingFace Trainer with the given configuration."""
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to=config.report_to,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    return trainer
