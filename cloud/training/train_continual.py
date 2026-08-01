#!/usr/bin/env python3
"""
Continual Training Script for DeepSeek Fine-Tuning.
Loads an existing trained adapter, streams the next 200k dataset slice,
and continues QLoRA fine-tuning with production best practices & maximum efficiency.
"""

import os
import sys
import argparse
import yaml
import json
import logging
import warnings
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DefaultDataCollator
    )
    from peft import (
        PeftModel,
        PeftConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    from datasets import Dataset
    import ijson
    import psutil
except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    sys.exit(1)


def setup_logging(log_dir: str, experiment_name: str):
    """Setup logging for continual training run."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("continual_train")
    return logger


def load_tokenizer(model_name: str, cache_dir: str, logger: logging.Logger):
    """Load and prepare DeepSeek tokenizer."""
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    return tokenizer


def load_model_and_adapter(base_model_name: str, adapter_path: str, cache_dir: str, logger: logging.Logger):
    """Load base 4-bit model and attach/resume trained adapter for continual training."""
    logger.info("Setting up 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    logger.info(f"Loading base model: {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False  # Critical for gradient checkpointing

    # Prepare model for k-bit training with gradient checkpointing enabled
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"✓ Loading pre-trained adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        model.config.use_cache = False
    else:
        logger.warning(f"Adapter path '{adapter_path}' not found! Initializing new LoRA adapter...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model ready. Trainable params: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    return model


def tokenize_conversations(examples, tokenizer, max_length=1024):
    """Tokenize conversation turns into input_ids, attention_mask, and labels."""
    input_ids_list, attention_mask_list, labels_list = [], [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for conversations in examples.get('conversations', [[]]):
        text_parts = []
        if isinstance(conversations, list):
            for turn in conversations:
                role = turn.get('role') or turn.get('from', '')
                content = turn.get('content') or turn.get('value', '')
                if role in ['user', 'human']:
                    text_parts.append(f"<|User|>: {content}\n")
                elif role in ['assistant', 'gpt']:
                    text_parts.append(f"<|Assistant|>: {content}\n")
        else:
            text_parts.append(str(conversations))

        text = ''.join(text_parts)
        tokenized = tokenizer(text, truncation=True, max_length=max_length)
        ids = tokenized['input_ids']
        length = len(ids)

        if length < max_length:
            input_ids = ids + [pad_id] * (max_length - length)
            attention_mask = [1] * length + [0] * (max_length - length)
            labels = ids + [-100] * (max_length - length)
        else:
            input_ids = ids[:max_length]
            attention_mask = [1] * max_length
            labels = ids[:max_length]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }


def load_next_dataset_slice(json_path: str, skip_samples: int, num_samples: int, logger: logging.Logger):
    """Stream dataset and skip previous samples to extract the next slice in memory."""
    logger.info(f"Streaming dataset from {json_path}...")
    logger.info(f"Skipping first {skip_samples} samples, taking next {num_samples} samples...")

    data_slice = []
    end_index = skip_samples + num_samples

    with open(json_path, 'r', encoding='utf-8') as f:
        parser = ijson.items(f, 'item', use_float=True)
        for i, item in enumerate(parser):
            if i < skip_samples:
                continue
            if i >= end_index:
                break
            data_slice.append(item)
            if len(data_slice) % 25000 == 0 and len(data_slice) > 0:
                logger.info(f"  Loaded {len(data_slice)} / {num_samples} samples...")

    logger.info(f"✓ Dataset slice loaded: {len(data_slice)} samples (indices {skip_samples} to {skip_samples + len(data_slice)})")
    dataset = Dataset.from_list(data_slice)
    del data_slice
    gc.collect()
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Continual Fine-Tuning on Next Dataset Slice')
    parser.add_argument('--config', type=str, default='config/cloud_full.yaml', help='YAML config file')
    parser.add_argument('--adapter-path', type=str, default='./outputs/final_adapter', help='Path to pre-trained adapter directory')
    parser.add_argument('--skip-samples', type=int, default=100000, help='Number of already trained samples to skip')
    parser.add_argument('--num-samples', type=int, default=200000, help='Number of new samples to train on')
    parser.add_argument('--batch-size', type=int, default=2, help='Per device batch size (2 for max GPU efficiency on T4)')
    parser.add_argument('--grad-accum', type=int, default=2, help='Gradient accumulation steps (2 for fast steps)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum training steps (1000 steps = 1h 45m limit)')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate for continual tuning')
    parser.add_argument('--output-name', type=str, default='deepseek_continual_200k', help='Experiment output name')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of CPU workers for tokenization')

    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    io_config = config.get('cloud', {}).get('io') or config.get('io', {})
    log_dir = io_config.get('log_dir', './logs')
    logger = setup_logging(log_dir, args.output_name)

    logger.info("=" * 60)
    logger.info("CONTINUAL FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Adapter Path: {args.adapter_path}")
    logger.info(f"Skip Samples: {args.skip_samples}")
    logger.info(f"Num Samples: {args.num_samples}")
    logger.info(f"Batch Size: {args.batch_size}, Grad Accum: {args.grad_accum}")

    base_model = config.get('model', {}).get('base_model', 'deepseek-ai/deepseek-coder-6.7b-instruct')
    cache_dir = io_config.get('cache_dir', './cache')

    # Load Tokenizer & Model + Adapter
    tokenizer = load_tokenizer(base_model, cache_dir, logger)
    model = load_model_and_adapter(base_model, args.adapter_path, cache_dir, logger)

    # Load next dataset slice
    merged_dir = io_config.get('dataset_dir', {}).get('merged', './datasets/merged')
    train_json = os.path.join(merged_dir, 'train.json')
    val_json = os.path.join(merged_dir, 'validation.json')

    if not os.path.exists(train_json):
        logger.error(f"Dataset file not found at {train_json}")
        return 1

    train_raw = load_next_dataset_slice(train_json, args.skip_samples, args.num_samples, logger)

    num_proc = min(args.num_workers, os.cpu_count() or 1)
    logger.info(f"Fast parallel tokenizing ({num_proc} CPU processes)...")
    train_dataset = train_raw.map(
        lambda x: tokenize_conversations(x, tokenizer, args.max_seq_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=train_raw.column_names,
        desc="Tokenizing 200k slice"
    )

    val_dataset = None
    if os.path.exists(val_json):
        val_raw = load_next_dataset_slice(val_json, skip_samples=0, num_samples=2500, logger=logger)
        val_dataset = val_raw.map(
            lambda x: tokenize_conversations(x, tokenizer, args.max_seq_length),
            batched=True,
            num_proc=num_proc,
            remove_columns=val_raw.column_names,
            desc="Tokenizing validation set"
        )

    # Configure training arguments
    checkpoint_dir = os.path.join(io_config.get('checkpoint_dir', './checkpoints'), args.output_name)
    output_dir = os.path.join(io_config.get('output_dir', './outputs'), args.output_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Memory fragmentation optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=25,
        save_steps=200,
        eval_steps=200 if val_dataset else 0,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        disable_tqdm=False,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        save_total_limit=3,
        report_to=["tensorboard"],
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DefaultDataCollator(return_tensors="pt")
    )

    logger.info("🚀 Starting continual training on next 200k dataset slice...")
    result = trainer.train()

    final_adapter_path = os.path.join(output_dir, "final_continual_adapter")
    logger.info(f"Saving final continual adapter to {final_adapter_path}...")
    trainer.save_model(final_adapter_path)

    logger.info("=" * 60)
    logger.info("✓ CONTINUAL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"  Final adapter: {final_adapter_path}")
    logger.info(f"  Final Training Loss: {result.training_loss:.4f}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
