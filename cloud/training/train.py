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
import time
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
    import ijson
except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    print("Install with: pip install transformers peft bitsandbytes accelerate datasets torch evaluate ijson")
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
        model_name = config.get('model', {}).get('base_model') or 'deepseek-ai/deepseek-coder-6.7b-instruct'
        if not model_name:
            logger.info("Using default model: deepseek-ai/deepseek-coder-6.7b-instruct")
            model_name = 'deepseek-ai/deepseek-coder-6.7b-instruct'
        logger.info(f"Model: {model_name}")
    
    # Check tokenized dataset - but we use raw JSON so skip this check
    if validation_config.get('check_dataset', True):
        logger.info("Using raw JSON datasets directly - tokenization check skipped")
    
    logger.info("✓ Environment validation passed")
    return True


def load_tokenizer(model_config: Dict[str, Any], cache_dir: str, logger: logging.Logger):
    """Load the DeepSeek tokenizer."""
    try:
        logger.info("Loading tokenizer...")
        
        # Get model name with fallback
        model_name = model_config.get('base_model') or 'deepseek-ai/deepseek-coder-6.7b-instruct'
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision=model_config.get('model_revision', 'main')
        )
        
        # Configure special tokens
        special_tokens = model_config.get('special_tokens', {})
        
        # Set pad token - use EOS if custom pad token doesn't exist
        if tokenizer.pad_token is None:
            if special_tokens.get('pad_token') and special_tokens['pad_token'] in tokenizer.get_vocab():
                tokenizer.pad_token = special_tokens['pad_token']
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(special_tokens['pad_token'])
            else:
                # Use EOS token as pad token (common for LLaMA-style models)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.info(f"✓ Tokenizer loaded: {len(tokenizer)} vocab size")
        logger.info(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        logger.info(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None


def load_model(model_config: Dict[str, Any], quantization_config: Dict[str, Any], 
              cache_dir: str, logger: logging.Logger):
    """Load and prepare the DeepSeek model with quantization."""
    try:
        logger.info("Loading model with quantization...")
        
        # Get model name with fallback
        model_name = model_config.get('base_model') or 'deepseek-ai/deepseek-coder-6.7b-instruct'
        
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
            model_name,
            quantization_config=bnb_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision=model_config.get('model_revision', 'main'),
            torch_dtype=getattr(torch, model_config.get('loading', {}).get('torch_dtype', 'bfloat16')),
            device_map="auto",
            low_cpu_mem_usage=model_config.get('loading', {}).get('low_cpu_mem_usage', True)
        )
        
        logger.info(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        model.config.use_cache = False  # Disable KV cache for training with gradient checkpointing
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def setup_lora(model: nn.Module, lora_config: Dict[str, Any], logger: logging.Logger):
    """Setup LoRA adaptation layer."""
    try:
        logger.info("Setting up LoRA...")
        
        # Prepare model for k-bit training with gradient checkpointing enabled
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
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


def tokenize_conversation_batch(examples, tokenizer, max_length):
    """Tokenize a batch of conversation examples."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for conversations in examples['conversations']:
        # Build text from conversation with user prompts and assistant responses
        text_parts = []
        assistant_start_indices = []
        
        for turn in conversations:
            role = turn.get('role') or turn.get('from', '')
            content = turn.get('content') or turn.get('value', '')
            
            if role in ['user', 'human']:
                text_parts.append(f"<|User|>: {content}\n")
            elif role in ['assistant', 'gpt']:
                text_parts.append(f"<|Assistant|>: {content}\n")
        
        text = ''.join(text_parts)
        
        # Tokenize without padding first to get actual length
        tokenized_no_pad = tokenizer(text, truncation=True, max_length=max_length)
        input_ids = tokenized_no_pad['input_ids']
        
        # Find where assistant responses are - we want to mask user prompts in labels
        # For simplicity, we'll use the full sequence but mask only padding
        # In causal LM, the model learns to predict from all tokens
        
        labels = input_ids.copy()
        
        # Pad to max_length
        actual_length = len(input_ids)
        if actual_length < max_length:
            input_ids = input_ids + [pad_token_id] * (max_length - actual_length)
            attention_mask = [1] * actual_length + [0] * (max_length - actual_length)
            # Replace padding in labels with -100
            labels = labels + [-100] * (max_length - actual_length)
        else:
            attention_mask = [1] * max_length
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }


def load_tokenized_datasets(io_config: Dict[str, Any], model_config: Dict[str, Any], 
                           tokenize_config: Dict[str, Any], tokenizer, logger: logging.Logger):
    """Load training and validation datasets - tries raw JSON first, then tokenized."""
    try:
        # Check if we can use raw JSON datasets (bypasses tokenization)
        merged_dir = io_config.get('dataset_dir', {}).get('merged', './datasets/merged')
        train_json = os.path.join(merged_dir, 'train.json')
        val_json = os.path.join(merged_dir, 'validation.json')
        
        if os.path.exists(train_json):
            logger.info(f"Loading raw JSON datasets from {merged_dir}...")
            logger.info("Using memory-efficient streaming loading...")
            
            import json as json_lib
            import ijson
            
            # Sample limits - load only what we need
            max_training_samples = 100000  # 100k training samples (reduced from 500k for T4)
            max_validation_samples = 5000    # 5k validation samples (reduced from 10k)
            
            train_data = []
            file_size_mb = os.path.getsize(train_json) / (1024 * 1024)
            logger.info(f"Training file size: {file_size_mb:.1f} MB - loading {max_training_samples} samples max")
            
            # Use ijson for streaming JSON array parsing
            logger.info(f"Streaming JSON array with sample limit: {max_training_samples}")
            with open(train_json, 'r', encoding='utf-8') as f:
                parser = ijson.items(f, 'item', use_float=True)
                for i, item in enumerate(parser):
                    if i >= max_training_samples:
                        logger.info(f"Reached training sample limit: {max_training_samples}")
                        break
                    train_data.append(item)
                    if i % 10000 == 0 and i > 0:
                        logger.info(f"  Loaded {i} training samples...")
            
            logger.info(f"✓ Loaded {len(train_data)} training samples")
            
            # Load validation data
            val_data = []
            if os.path.exists(val_json):
                val_file_size = os.path.getsize(val_json) / (1024 * 1024)
                logger.info(f"Validation file size: {val_file_size:.1f} MB - loading {max_validation_samples} samples max")
                
                with open(val_json, 'r', encoding='utf-8') as f:
                    parser = ijson.items(f, 'item', use_float=True)
                    for i, item in enumerate(parser):
                        if i >= max_validation_samples:
                            logger.info(f"Reached validation sample limit: {max_validation_samples}")
                            break
                        val_data.append(item)
                
                logger.info(f"✓ Loaded {len(val_data)} validation samples")
            
            logger.info(f"✓ Final training data: {len(train_data)} samples")
            logger.info(f"✓ Final validation data: {len(val_data)} samples")
            
            # Monitor memory during dataset creation
            import gc
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024**3)
            logger.info(f"💾 Memory before dataset creation: {mem_before:.2f}GB")
            
            # Convert to datasets and cleanup
            train_dataset = Dataset.from_list(train_data)
            del train_data  # Free memory immediately
            gc.collect()
            
            val_dataset = Dataset.from_list(val_data) if val_data else None
            if val_data:
                del val_data
            gc.collect()
            
            mem_after = process.memory_info().rss / (1024**3)
            logger.info(f"💾 Memory after dataset creation: {mem_after:.2f}GB")
            
            # Tokenize datasets on-the-fly
            logger.info("Tokenizing datasets...")
            max_length = tokenize_config.get('max_seq_length', 2048)
            
            train_dataset = train_dataset.map(
                lambda x: tokenize_conversation_batch(x, tokenizer, max_length),
                batched=True,
                remove_columns=['conversations'],
                desc="Tokenizing training data"
            )
            logger.info(f"✓ Training dataset tokenized: {len(train_dataset)} samples")
            
            if val_dataset:
                val_dataset = val_dataset.map(
                    lambda x: tokenize_conversation_batch(x, tokenizer, max_length),
                    batched=True,
                    remove_columns=['conversations'],
                    desc="Tokenizing validation data"
                )
                logger.info(f"✓ Validation dataset tokenized: {len(val_dataset)} samples")
            
            logger.info("✓ Datasets loaded and tokenized from raw JSON")
            
            if mem_after > 12:
                logger.warning(f"⚠️ High memory usage after dataset loading: {mem_after:.2f}GB")
            
            return train_dataset, val_dataset
        
        # Fallback to tokenized datasets if raw JSON not available
        logger.info("Raw JSON not found, trying tokenized datasets...")
        tokenized_dir = io_config.get('dataset_dir', {}).get('tokenized', './datasets/tokenized')
        
        train_path = os.path.join(tokenized_dir, 'train')
        val_path = os.path.join(tokenized_dir, 'validation')
        
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
        logger.info("Creating Trainer with progress monitoring...")
        
        # Add progress callback
        from transformers import TrainerCallback
        import psutil
        import torch
        
        class ProgressCallback(TrainerCallback):
            def __init__(self, total_samples, max_memory_gb=14):
                self.total_samples = total_samples
                self.start_time = None
                self.max_memory_gb = max_memory_gb
                self.max_steps = None
            
            def get_memory_usage(self):
                """Get current memory usage in GB"""
                process = psutil.Process()
                mem_info = process.memory_info()
                ram_gb = mem_info.rss / (1024**3)  # Resident Set Size in GB
                
                # GPU memory if available
                gpu_gb = 0
                if torch.cuda.is_available():
                    gpu_gb = torch.cuda.memory_allocated() / (1024**3)
                
                return ram_gb, gpu_gb
            
            def log_memory(self, context=""):
                """Log memory usage and optimize if approaching limit"""
                ram_gb, gpu_gb = self.get_memory_usage()
                total_gb = ram_gb + gpu_gb
                
                logger.info(f"💾 Memory [{context}]: RAM={ram_gb:.2f}GB, GPU={gpu_gb:.2f}GB, Total={total_gb:.2f}GB (Limit: {self.max_memory_gb}GB)")
                
                if total_gb > self.max_memory_gb * 0.95:
                    logger.warning(f"⚠️ Memory usage high: {total_gb:.2f}GB used")
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                elif total_gb > self.max_memory_gb:
                    logger.warning(f"⚠️ Memory total ({total_gb:.2f}GB) exceeds soft threshold ({self.max_memory_gb}GB) - running gc")
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            def on_train_begin(self, args, state, control, **kwargs):
                self.start_time = time.time()
                self.max_steps = args.max_steps if args.max_steps > 0 else state.max_steps
                logger.info(f"🚀 Training started with {self.total_samples} samples")
                logger.info(f"📊 Training configuration: {args.num_train_epochs} epochs, {self.max_steps} max steps, batch size {args.per_device_train_batch_size}")
                logger.info(f"⏱️ Estimated completion time: {self.max_steps / 2:.1f} minutes (assuming ~30 sec/step)")
                logger.info(f"💾 Memory limit: {self.max_memory_gb}GB")
                self.log_memory("start")
            
            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step % 10 == 0:
                    logger.info(f"🔄 Step {state.global_step}: Processing training data...")
                    self.log_memory(f"step_{state.global_step}")
            
            def on_step_end(self, args, state, control, **kwargs):
                if state.log_history:
                    last_log = state.log_history[-1]
                    if 'loss' in last_log:
                        elapsed_time = time.time() - self.start_time
                        steps_per_sec = state.global_step / elapsed_time if elapsed_time > 0 else 0
                        eta = (state.max_steps - state.global_step) / steps_per_sec if steps_per_sec > 0 and state.max_steps else 0
                        logger.info(f"📈 Step {state.global_step}: loss={last_log['loss']:.4f}, speed={steps_per_sec:.2f} steps/s, ETA={eta/60:.1f} min")
            
            def on_epoch_end(self, args, state, control, **kwargs):
                logger.info(f"✅ Epoch {state.epoch} completed")
                self.log_memory(f"epoch_{state.epoch}")
            
            def on_train_end(self, args, state, control, **kwargs):
                elapsed_time = time.time() - self.start_time
                logger.info(f"🎉 Training completed in {elapsed_time/60:.1f} minutes")
                self.log_memory("final")
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                self.log_memory("evaluation_start")
                if metrics:
                    logger.info(f"📊 Evaluation: {metrics}")
                self.log_memory("evaluation_end")
        
        # Get output paths
        output_dir = io_config.get('output_dir', './outputs')
        adapter_dir = io_config.get('adapter_dir', './adapters')
        checkpoint_dir = io_config.get('checkpoint_dir', './checkpoints')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-detect hardware bfloat16 support (T4 GPUs do not support bf16)
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_bf16 = training_config.get('bf16', False) and bf16_supported
        use_fp16 = training_config.get('fp16', True) if not use_bf16 else False
        
        if training_config.get('bf16', False) and not bf16_supported:
            logger.warning("⚠️ Hardware does not natively support bfloat16 (e.g. NVIDIA T4). Automatically falling back to fp16=True.")

        eval_strat = "steps" if val_dataset is not None else "no"
        load_best = training_config.get('load_best_model_at_end', True) if val_dataset is not None else False

        # Create training arguments with progress bars enabled
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
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=training_config['gradient_checkpointing'],
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            max_grad_norm=training_config['max_grad_norm'],
            weight_decay=training_config['weight_decay'],
            optim=training_config['optim'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            disable_tqdm=False,  # Force enable progress bars
            load_best_model_at_end=load_best,
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            save_total_limit=training_config.get('save_total_limit', 3),
            report_to=training_config.get('report_to', []),
            seed=training_config.get('seed', 42),
            eval_strategy=eval_strat,
            save_strategy="steps",
            logging_first_step=True,
            dataloader_pin_memory=False,  # Reduce memory
            gradient_checkpointing_kwargs={'use_reentrant': False}  # Avoid reentrant bugs
        )
        # Use default data collator since dataset is already tokenized with input_ids and labels
        from transformers import DefaultDataCollator
        data_collator = DefaultDataCollator(return_tensors="pt")
        
        # Debug: Check dataset format
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"📋 Sample dataset keys: {list(sample.keys())}")
            logger.info(f"📋 input_ids shape: {len(sample.get('input_ids', []))}")
            logger.info(f"📋 attention_mask shape: {len(sample.get('attention_mask', []))}")
            logger.info(f"📋 labels shape: {len(sample.get('labels', []))}")
        
        # Create trainer with progress and memory monitoring callback
        max_memory_gb = 14  # 14GB RAM limit
        # Enable efficient CUDA memory allocation and prevent fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback(len(train_dataset), max_memory_gb)]
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


def train_model(trainer, json_log_file: str, logger: logging.Logger, resume_from_checkpoint: str = None):
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
        
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
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
    
    # Get configurations with fallbacks
    io_config = config.get('cloud', {}).get('io') or config.get('io', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    lora_config = training_config.get('lora', {})
    quantization_config = model_config.get('quantization', {})
    
    # Setup logging
    log_dir = io_config.get('log_dir', './logs')
    logger, json_log_file = setup_logging(log_dir, args.output_name)
    
    logger.info("=" * 60)
    logger.info("DeepSeek Fine-tuning Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Output name: {args.output_name}")
    
    # Save training configuration - ensure output directory exists
    output_dir = io_config.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    config_output = os.path.join(output_dir, f"{args.output_name}_config.json")
    save_training_config(config, config_output)
    
    # Validate environment
    if not validate_environment(config, logger):
        logger.error("Environment validation failed")
        return 1
    
    # Load tokenizer
    cache_dir = io_config.get('cache_dir', './cache') if io_config else './cache'
    tokenizer = load_tokenizer(model_config, cache_dir, logger)
    if tokenizer is None:
        return 1
    
    # Load model
    model = load_model(model_config, quantization_config, cache_dir, logger)
    if model is None:
        return 1
    
    # Setup LoRA
    model = setup_lora(model, lora_config, logger)
    if model is None:
        return 1
    
    # Load datasets (will try raw JSON first, avoiding tokenization)
    model_config = config.get('model', {})
    tokenize_config = config.get('dataset', {}).get('tokenization', {})
    train_dataset, val_dataset = load_tokenized_datasets(io_config, model_config, tokenize_config, tokenizer, logger)
    if train_dataset is None:
        return 1
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, 
                            training_config, io_config or {}, logger)
    if trainer is None:
        return 1
    
    # Resume from checkpoint if specified
    resume_path = None
    if args.resume or args.checkpoint:
        if args.checkpoint:
            resume_path = args.checkpoint
        else:
            resume_path = find_latest_checkpoint(io_config.get('checkpoint_dir', './checkpoints'), logger)
        
        if resume_path:
            logger.info(f"Found checkpoint for resume: {resume_path}")
        else:
            logger.warning("No checkpoint found for resume, starting from scratch")
    
    # Train model
    result, final_adapter = train_model(trainer, json_log_file, logger, resume_from_checkpoint=resume_path)
    
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