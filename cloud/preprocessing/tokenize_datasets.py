#!/usr/bin/env python3
"""
Tokenize merged datasets for efficient training.
Supports automatic sequence length handling and parallel processing.
"""

import os
import sys
import argparse
import yaml
import json
import logging as standard_logging
from pathlib import Path
from tqdm import tqdm
import torch

try:
    from transformers import AutoTokenizer
    from datasets import Dataset
    from multiprocessing import cpu_count
except ImportError:
    standard_logging.error("Error: Required libraries not installed")
    standard_logging.error("Install with: pip install transformers datasets torch numpy")
    sys.exit(1)


def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_file} not found")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in {config_file}: {e}")
        return None


def load_tokenizer(model_config, cache_dir=None):
    """Load the DeepSeek tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['base_model'],
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision=model_config.get('model_revision', 'main')
        )
        
        # Configure special tokens if specified
        special_tokens = model_config.get('special_tokens', {})
        if special_tokens.get('pad_token'):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = special_tokens['pad_token']
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(special_tokens['pad_token'])
        
        if special_tokens.get('eos_token'):
            if tokenizer.eos_token is None:
                tokenizer.eos_token = special_tokens['eos_token']
        
        print(f"✓ Loaded tokenizer: {model_config['base_model']}")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Max length: {tokenizer.model_max_length}")
        
        return tokenizer
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return None


def load_merged_dataset(train_path, val_path, max_samples=None):
    """Load merged training and validation datasets with memory limits."""
    try:
        print("Loading merged datasets...")
        
        train_data = []
        if train_path and os.path.exists(train_path):
            with open(train_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  Found {len(data)} training samples in file")
                if max_samples:
                    print(f"  Limiting to {max_samples} samples to prevent OOM")
                    train_data = data[:max_samples]
                else:
                    train_data = data
            print(f"  Loaded {len(train_data)} training samples")
        
        val_data = []
        if val_path and os.path.exists(val_path):
            with open(val_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  Found {len(data)} validation samples in file")
                if max_samples:
                    val_max = max_samples // 10  # Validation is 10% of training
                    print(f"  Limiting to {val_max} samples to prevent OOM")
                    val_data = data[:val_max]
                else:
                    val_data = data
            print(f"  Loaded {len(val_data)} validation samples")
        
        if not train_data:
            print("✗ No training data found")
            return None, None
        
        return train_data, val_data
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        return None, None


def format_conversations_for_tokenizer(conversations):
    """Format conversations into a single string for tokenization."""
    if not conversations or not isinstance(conversations, list):
        return ""
    
    formatted = []
    for msg in conversations:
        if isinstance(msg, dict):
            role = msg.get('from', 'human')
            content = msg.get('value', '')
            
            if role == 'human':
                formatted.append(f"<|User|>: {content}")
            elif role == 'gpt':
                formatted.append(f"<|Assistant|>: {content}")
    
    return "\n".join(formatted)


def tokenize_function(examples, tokenizer, max_seq_length, tokenize_config):
    """Tokenize dataset examples."""
    texts = []
    
    for conversations in examples['conversations']:
        if isinstance(conversations, list):
            text = format_conversations_for_tokenizer(conversations)
        else:
            text = str(conversations)
        
        # Add system prompt if needed
        if not text.startswith("<|User|>"):
            text = f"<|User|>: {text}"
        
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=tokenize_config.get('truncation', True),
        padding=tokenize_config.get('padding', 'max_length'),
        return_attention_mask=tokenize_config.get('return_attention_mask', True),
        return_token_type_ids=tokenize_config.get('return_token_type_ids', False)
    )
    
    # Create labels for causal language modeling
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    # Pad token labels should be -100 to ignore in loss
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized['labels'] = [
        [(lbl if lbl != pad_token_id else -100) for lbl in labels]
        for labels in tokenized['labels']
    ]
    
    return tokenized


def analyze_tokenization(dataset, tokenizer):
    """Analyze tokenization statistics."""
    print("Analyzing tokenization...")
    
    if len(dataset) == 0:
        return
    
    # Sample some examples for statistics
    sample_size = min(1000, len(dataset))
    sample_indices = list(range(sample_size))
    
    lengths = []
    total_tokens = 0
    padding_tokens = 0
    
    for idx in tqdm(sample_indices, desc="Analyzing tokenization"):
        example = dataset[idx]
        
        # Count non-padding tokens
        input_ids = example['input_ids']
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        non_pad_count = sum(1 for token in input_ids if token != pad_token_id)
        pad_count = len(input_ids) - non_pad_count
        
        lengths.append(non_pad_count)
        total_tokens += non_pad_count
        padding_tokens += pad_count
    
    print(f"Tokenization Statistics (sampled {sample_size} examples):")
    print(f"  Average sequence length: {sum(lengths) / len(lengths):.1f} tokens")
    print(f"  Max sequence length: {max(lengths) if lengths else 0} tokens")
    print(f"  Min sequence length: {min(lengths) if lengths else 0} tokens")
    print(f"  Total tokens (sampled): {total_tokens}")
    print(f"  Padding tokens (sampled): {padding_tokens} ({padding_tokens / (total_tokens + padding_tokens) * 100:.1f}%)")


def save_tokenized_dataset(dataset, output_path, dataset_type="train"):
    """Save tokenized dataset."""
    try:
        print(f"Saving {dataset_type} dataset to {output_path}...")
        dataset.save_to_disk(output_path)
        print(f"✓ Saved {dataset_type} dataset ({len(dataset)} samples)")
        return True
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        return False


def estimate_dataset_size(dataset):
    """Estimate tokenized dataset size on disk."""
    try:
        # Count total tokens
        total_tokens = 0
        for example in dataset:
            input_ids = example['input_ids']
            # Count non-padding tokens
            pad_token_id = 0  # Default, might need adjustment
            non_pad = sum(1 for token in input_ids if token != pad_token_id)
            total_tokens += non_pad
        
        # Estimate memory (rough estimate)
        # Each token is typically 2 bytes (int16) plus overhead
        estimated_gb = (total_tokens * 4) / 1024 / 1024 / 1024
        
        return total_tokens, estimated_gb
    except Exception as e:
        print(f"Error estimating size: {e}")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description='Tokenize datasets for training')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--train-file', type=str, default=None,
                        help='Training dataset file')
    parser.add_argument('--val-file', type=str, default=None,
                        help='Validation dataset file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for tokenized data')
    parser.add_argument('--max-seq-length', type=int, default=None,
                        help='Maximum sequence length')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of workers for processing')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip tokenization analysis')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Tokenization")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get configurations
    model_config = config.get('model', {})
    if not model_config:
        model_config = {
            'base_model': 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'model_revision': 'main',
            'trust_remote_code': True,
            'special_tokens': {
                'pad_token': '<|pad|>',
                'eos_token': '<|end_of_sentence|>'
            }
        }
    tokenize_config = config.get('dataset', {}).get('tokenization', {})
    io_config = config.get('cloud', {}).get('io') or config.get('io', {})
    
    # Determine parameters
    max_seq_length = args.max_seq_length or tokenize_config.get('max_seq_length', 4096)
    num_workers = args.num_workers or min(cpu_count(), 2)  # Reduce workers to prevent OOM
    cache_dir = io_config.get('cache_dir', './cache')
    
    print(f"Max sequence length: {max_seq_length}")
    print(f"Number of workers: {num_workers}")
    print(f"Cache directory: {cache_dir}")
    
    # Determine file paths
    if args.train_file:
        train_path = args.train_file
    else:
        train_path = os.path.join(io_config['dataset_dir']['merged'], 'train.json')
    
    if args.val_file:
        val_path = args.val_file
    else:
        val_path = os.path.join(io_config['dataset_dir']['merged'], 'validation.json')
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = io_config['dataset_dir']['tokenized']
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Output directory: {output_dir}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_config, cache_dir)
    if tokenizer is None:
        return 1
    
    # Load datasets with memory limits
    # Force small dataset for testing to prevent OOM kills
    max_samples = 1000  # Start with very small sample to test tokenization
    train_data, val_data = load_merged_dataset(train_path, val_path, max_samples=max_samples)
    if train_data is None:
        return 1
    
    # Create datasets
    print("Converting to Hugging Face datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None
    
    # Process training dataset
    print(f"\nTokenizing training dataset...")
    tokenized_train = train_dataset.map(
        function=lambda examples: tokenize_function(examples, tokenizer, max_seq_length, tokenize_config),
        batched=True,
        batch_size=100,  # Smaller batches to prevent OOM
        num_proc=num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    # Process validation dataset
    if val_dataset:
        print(f"\nTokenizing validation dataset...")
        tokenized_val = val_dataset.map(
            function=lambda examples: tokenize_function(examples, tokenizer, max_seq_length, tokenize_config),
            batched=True,
            batch_size=100,  # Smaller batches to prevent OOM
            num_proc=num_workers,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation"
        )
    else:
        tokenized_val = None
    
    # Analyze tokenization
    if not args.skip_analysis:
        analyze_tokenization(tokenized_train, tokenizer)
        if tokenized_val:
            analyze_tokenization(tokenized_val, tokenizer)
    
    # Save tokenized datasets
    train_output = os.path.join(output_dir, 'train')
    val_output = os.path.join(output_dir, 'validation')
    
    if not save_tokenized_dataset(tokenized_train, train_output, "train"):
        return 1
    
    if tokenized_val and not save_tokenized_dataset(tokenized_val, val_output, "validation"):
        return 1
    
    # Save metadata
    metadata = {
        'tokenizer': model_config['base_model'],
        'max_seq_length': max_seq_length,
        'train_samples': len(tokenized_train),
        'val_samples': len(tokenized_val) if tokenized_val else 0,
        'num_workers': num_workers
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Tokenization complete!")
    print(f"  Training data: {train_output}")
    print(f"  Validation data: {val_output}")
    print(f"  Metadata: {metadata_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())