#!/usr/bin/env python3
"""
Main training script for DeepSeek fine-tuning pipeline.
Production-ready QLoRA training with full dataset support and automatic resume.
"""

import os
import sys
import argparse
import yaml
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Suppress specific warnings
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
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig, 
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
    import pandas as pd
    from tqdm import tqdm
    import evaluate
except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    print("Install with: pip install transformers peft bitsandbytes accelerate datasets torch evaluate")
    sys.exit(1)


def save_training_config(config_dict: Dict[str, Any], output_path: str):
    """Save training configuration to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
        logging.info(f"Training configuration saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving training config: {e}")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_file}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_file} not found")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Invalid YAML in {config_file}: {e}")
        return None


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup comprehensive logging for training."""
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create JSON log file for metrics
    json_log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}_metrics.json")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Metrics JSON: {json_log_file}")
    
    return logger, json_log_file


def validate_environment(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validate training environment before starting."""
    validation_config = config.get('cloud', {}).get('validation', {})
    
    logger.info("Validating training environment...")
    
    # Check CUDA
    if validation_config.get('check_cuda', True):
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check VRAM
    if validation_config.get('check_vram', True):
        available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        required_vram = config.get('cloud', {}).get('available_vram', 16)
        
        logger.info(f"Available VRAM: {available_vram_gb:.1f} GB")
        logger.info(f"Required VRAM: {required_vram} GB")
        
        if available_vram_gb < required_vram * 0.8:  # 20% buffer
            logger.warning(f"Available VRAM ({available_vram_gb:.1f} GB) is less than recommended ({required_vram} GB)")
    
    # Check disk space
    if validation_config.get('min_disk_space', 0) > 0:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_space_gb = disk_usage.free / 1024**3
        min_required = validation_config.get('min_disk_space', 50)
        
        logger.info(f"Available disk space: {free_space_gb:.1f} GB")
        logger.info(f"Minimum required: {min_required} GB")
        
        if free_space_gb < min_required:
            logger.error(f"Insufficient disk space: {free_space_gb:.1f} GB < {min_required} GB")
            return False
    
    # Check model and tokenizer
    if validation_config.get('check_model', True):
        model_name = config.get('model', {}).get('base_model')
        if not model_name:
            logger.error("Model not specified in configuration")
            return False
        logger.info(f"Model: {model_name}")
    
    # Check tokenized dataset
    if validation_config.get('check_dataset', True):
        tokenized_path = os.path.join(
            config.get('cloud', {}).get('io', {}).get('dataset_dir', {}).get('tokenized', './datasets/tokenized'),
            'train'
        )
        if not os.path.exists(tokenized_path):
            logger.error(f"Tokenized dataset not found at {tokenized_path}")
            logger.error("Please run preprocessing/tokenize.py first")
            return False
        logger.info(f"Tokenized dataset found at {tokenized_path}")
    
    logger.info("✓ Environment validation passed")
    return True


def load_tokenizer(model_config: Dict[str, Any], cache_dir: str, logger: logging.Logger):
    """Load the DeepSeek tokenizer."""
    try:
        logger.info("Loading tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['base_model'],
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision=model_config.get('model_revision', 'main')
        )
        
        # Configure special tokens
        special_tokens = model_config.get('special_tokens', {})
        if special_tokens.get('pad_token'):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = special_tokens['pad_token']
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(special_tokens['pad_token'])
        
        logger.info(f"✓ Tokenizer loaded: {len(tokenizer)} vocab size")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None


def load_model(model_config: Dict[str, Any], quantization_config: Dict[str, Any], 
              cache_dir: str, logger: logging.Logger):
    """Load and prepare the DeepSeek model with quantization."""
    try:
        logger.info("Loading model with quantization...")
        
        # Create BNB config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', True),
            load_in_8bit=quantization_config.get('load_in_8bit', False),
            bnb_4bit_compute_dtype=torch.bfloat16 if quantization_config.get('bnb_4bit_compute_dtype') == "bfloat16" else torch.float16,
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', "nf4"),
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True)
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config['base_model'],
            quantization_config=bnb_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision=model_config.get('model_revision', 'main'),
            torch_dtype=getattr(torch, model_config.get('loading', {}).get('torch_dtype', 'bfloat16')),
            device_map="auto",
            low_cpu_mem_usage=model_config.get('loading', {}).get('low_cpu_mem_usage', True)
        )
        
        logger.info(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def setup_lora(model: nn.Module, lora_config: Dict[str, Any], logger: logging.Logger):
    """Setup LoRA adaptation layer."""
    try:
        logger.info("Setting up LoRA...")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Create LoRA config
        lora = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj"]),
            bias=lora_config.get('bias', "none"),
            task_type=TaskType.CAUSAL_LM
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info(f"✓ LoRA setup complete")
        logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_percent:.2f}%)")
        
        # Print trainable modules
        logger.info("  Trainable modules:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"    {name}")
        
        return model
    except Exception as e:
        logger.error(f"Error setting up LoRA: {e}")
        return None


def load_tokenized_datasets(io_config: Dict[str, Any], logger: logging.Logger):
    """Load tokenized training and validation datasets."""
    try:
        tokenized_dir = io_config.get('dataset_dir', {}).get('tokenized', './datasets/tokenized')
        
        train_path = os.path.join(tokenized_dir, 'train')
        val_path = os.path.join(tokenized_dir, 'validation')
        
        logger.info(f"Loading datasets from {tokenized_dir}...")
        
        # Load training dataset
        if os.path.exists(train_path):
            train_dataset = Dataset.load_from_disk(train_path)
            logger.info(f"✓ Training dataset loaded: {len(train_dataset)} samples")
        else:
            logger.error(f"Training dataset not found at {train_path}")
            return None, None
        
        # Load validation dataset
        if os.path.exists(val_path):
            val_dataset = Dataset.load_from_disk(val_path)
            logger.info(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
        else:
            logger.warning("Validation dataset not found, using subset of training data")
            val_dataset = train_dataset.shuffle().select(range(min(1000, len(train_dataset))))
        
        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return None, None


def create_trainer(model, tokenizer, train_dataset, val_dataset, 
                  training_config: Dict[str, Any], io_config: Dict[str, Any],
                  logger: logging.Logger):
    """Create Trainer with custom callbacks."""
    try:
        logger.info("Creating Trainer...")
        
        # Get output paths
        output_dir = io_config.get('output_dir', './outputs')
        adapter_dir = io_config.get('adapter_dir', './adapters')
        checkpoint_dir = io_config.get('checkpoint_dir', './checkpoints')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', training_config['per_device_train_batch_size']),
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config.get('eval_steps', training_config['save_steps']),
            max_steps=training_config.get('max_steps', 0),
            fp16=training_config.get('fp16', False),
            bf16=training_config.get('bf16', True),
            gradient_checkpointing=training_config['gradient_checkpointing'],
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            max_grad_norm=training_config['max_grad_norm'],
            weight_decay=training_config['weight_decay'],
            optim=training_config['optim'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            disable_tqdm=training_config.get('disable_tqdm', False),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            save_total_limit=training_config.get('save_total_limit', 3),
            report_to=training_config.get('report_to', []),
            seed=training_config.get('seed', 42),
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        logger.info("✓ Trainer created successfully")
        logger.info(f"  Training epochs: {training_args.num_train_epochs}")
        logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        logger.info(f"  Warmup steps: {training_args.warmup_steps}")
        
        return trainer
    except Exception as e:
        logger.error(f"Error creating trainer: {e}")
        return None


def train_model(trainer, json_log_file: str, logger: logging.Logger):
    """Main training loop with comprehensive logging."""
    try:
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)
        
        # Initialize metrics tracking
        metrics_history = {
            'training_loss': [],
            'learning_rate': [],
            'eval_loss': [],
            'timestamp': [],
            'step': []
        }
        
        # Start training
        start_time = datetime.now()
        logger.info(f"Training start time: {start_time}")
        
        result = trainer.train()
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Training end time: {end_time}")
        logger.info(f"Total training duration: {training_duration / 3600:.2f} hours")
        
        # Log final metrics
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Final training loss: {result.training_loss}")
        logger.info(f"Steps trained: {result.global_step}")
        
        # Save final adapter
        logger.info("Saving final adapter...")
        final_adapter_path = os.path.join(trainer.args.output_dir, "final_adapter")
        trainer.save_model(final_adapter_path)
        logger.info(f"✓ Final adapter saved to {final_adapter_path}")
        
        return result, final_adapter_path
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def find_latest_checkpoint(checkpoint_dir: str, logger: logging.Logger) -> str:
    """Find the latest checkpoint for resuming."""
    try:
        if not os.path.exists(checkpoint_dir):
            return None
        
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        
        if not checkpoints:
            return None
        
        # Sort by checkpoint number
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoints[-1]
        
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        logger.error(f"Error finding checkpoint: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Train DeepSeek with QLoRA')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint to resume from')
    parser.add_argument('--output-name', type=str, default='deepseek_finetune',
                        help='Name for output files')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get configurations
    io_config = config['cloud']['io']
    model_config = config['model']
    training_config = config['training']
    lora_config = config['training']['lora']
    quantization_config = config['model']['quantization']
    
    # Setup logging
    logger, json_log_file = setup_logging(io_config['log_dir'], args.output_name)
    
    logger.info("=" * 60)
    logger.info("DeepSeek Fine-tuning Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Output name: {args.output_name}")
    
    # Save training configuration
    config_output = os.path.join(io_config['output_dir'], f"{args.output_name}_config.json")
    save_training_config(config, config_output)
    
    # Validate environment
    if not validate_environment(config, logger):
        logger.error("Environment validation failed")
        return 1
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_config, io_config['cache_dir'], logger)
    if tokenizer is None:
        return 1
    
    # Load model
    model = load_model(model_config, quantization_config, io_config['cache_dir'], logger)
    if model is None:
        return 1
    
    # Setup LoRA
    model = setup_lora(model, lora_config, logger)
    if model is None:
        return 1
    
    # Load datasets
    train_dataset, val_dataset = load_tokenized_datasets(io_config, logger)
    if train_dataset is None:
        return 1
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, 
                            training_config, io_config, logger)
    if trainer is None:
        return 1
    
    # Resume from checkpoint if specified
    if args.resume or args.checkpoint:
        if args.checkpoint:
            resume_path = args.checkpoint
        else:
            resume_path = find_latest_checkpoint(io_config['checkpoint_dir'], logger)
        
        if resume_path:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            trainer.train(resume_from_checkpoint=resume_path)
        else:
            logger.warning("No checkpoint found for resume, starting from scratch")
    
    # Train model
    result, final_adapter = train_model(trainer, json_log_file, logger)
    
    if result is None:
        logger.error("Training failed")
        return 1
    
    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Final adapter saved to: {final_adapter}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())