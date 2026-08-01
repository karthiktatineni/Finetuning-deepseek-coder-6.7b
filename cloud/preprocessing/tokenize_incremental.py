#!/usr/bin/env python3
"""
Incremental tokenization script that processes data in small batches to prevent OOM
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
        
        if not text.startswith("<|User|>"):
            text = f"<|User|>: {text}"
        
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=tokenize_config.get('truncation', True),
        padding=tokenize_config.get('padding', 'max_length'),
        return_attention_mask=tokenize_config.get('return_attention_mask', True),
        return_token_type_ids=tokenize_config.get('return_token_type_ids', False)
    )
    
    tokenized['labels'] = tokenized['input_ids'].copy()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized['labels'] = [
        [(lbl if lbl != pad_token_id else -100) for lbl in labels]
        for labels in tokenized['labels']
    ]
    
    return tokenized


def load_data_in_chunks(filepath, chunk_size=100):
    """Load JSON data in small chunks to prevent OOM."""
    def chunk_generator():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Total samples in file: {len(data)}")
                
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    print(f"Processing chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}")
                    yield chunk
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            yield []
    
    return chunk_generator()


def tokenize_in_chunks(filepath, tokenizer, output_dir, max_seq_length, tokenize_config, 
                       dataset_type="train", chunk_size=100, max_samples=None):
    """Tokenize data in small chunks and save incrementally."""
    print(f"\n{'='*60}")
    print(f"Tokenizing {dataset_type} dataset in chunks")
    print(f"{'='*60}")
    
    # Create output directory
    chunk_output_dir = os.path.join(output_dir, dataset_type)
    Path(chunk_output_dir).mkdir(parents=True, exist_ok=True)
    
    chunk_generator = load_data_in_chunks(filepath, chunk_size)
    
    all_tokenized = []
    total_samples = 0
    
    for chunk_idx, chunk in enumerate(chunk_generator):
        if not chunk:
            continue
            
        # Limit total samples if specified
        if max_samples and total_samples + len(chunk) > max_samples:
            chunk = chunk[:max_samples - total_samples]
        
        if not chunk:
            break
            
        print(f"  Chunk {chunk_idx + 1}: {len(chunk)} samples")
        
        # Convert to dataset
        chunk_dataset = Dataset.from_list(chunk)
        
        # Tokenize this chunk
        tokenized_chunk = chunk_dataset.map(
            function=lambda examples: tokenize_function(examples, tokenizer, max_seq_length, tokenize_config),
            batched=True,
            batch_size=50,
            num_proc=1,  # Single process to reduce memory
            remove_columns=chunk_dataset.column_names,
            desc=f"Tokenizing chunk {chunk_idx + 1}"
        )
        
        # Save this chunk
        chunk_path = os.path.join(chunk_output_dir, f"chunk_{chunk_idx:04d}")
        tokenized_chunk.save_to_disk(chunk_path)
        print(f"  ✓ Saved chunk to {chunk_path}")
        
        all_tokenized.append(tokenized_chunk)
        total_samples += len(tokenized_chunk)
        
        if max_samples and total_samples >= max_samples:
            break
        
        # Clear memory
        del chunk_dataset, tokenized_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"✓ Tokenized {total_samples} {dataset_type} samples in {len(all_tokenized)} chunks")
    return all_tokenized, total_samples


def save_metadata(output_dir, metadata):
    """Save tokenization metadata."""
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Incrementally tokenize datasets in chunks')
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
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum total samples to process')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Samples per chunk')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Incremental Dataset Tokenization (Memory-Safe)")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get configurations with defaults
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
    
    max_seq_length = args.max_seq_length or tokenize_config.get('max_seq_length', 4096)
    chunk_size = args.chunk_size
    max_samples = args.max_samples or 1000  # Default to 1000 samples for testing
    
    print(f"Max sequence length: {max_seq_length}")
    print(f"Chunk size: {chunk_size}")
    print(f"Max samples: {max_samples}")
    
    # Determine file paths
    if args.train_file:
        train_path = args.train_file
    else:
        train_path = os.path.join(io_config.get('dataset_dir', {}).get('merged', './datasets/merged'), 'train.json')
    
    if args.val_file:
        val_path = args.val_file
    else:
        val_path = os.path.join(io_config.get('dataset_dir', {}).get('merged', './datasets/merged'), 'validation.json')
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = io_config.get('dataset_dir', {}).get('tokenized', './datasets/tokenized')
    
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Output directory: {output_dir}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_config, io_config.get('cache_dir', './cache'))
    if tokenizer is None:
        return 1
    
    # Tokenize datasets incrementally
    train_chunks, train_samples = tokenize_in_chunks(
        train_path, tokenizer, output_dir, max_seq_length, tokenize_config,
        "train", chunk_size, max_samples
    )
    
    val_chunks = None
    val_samples = 0
    if os.path.exists(val_path):
        val_chunks, val_samples = tokenize_in_chunks(
            val_path, tokenizer, output_dir, max_seq_length, tokenize_config,
            "validation", chunk_size, max_samples // 10
        )
    
    # Save metadata
    metadata = {
        'tokenizer': model_config['base_model'],
        'max_seq_length': max_seq_length,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'train_chunks': len(train_chunks),
        'val_chunks': len(val_chunks) if val_chunks else 0,
        'chunk_size': chunk_size,
        'max_samples_limit': max_samples
    }
    
    save_metadata(output_dir, metadata)
    
    print(f"\n✓ Tokenization complete!")
    print(f"  Training: {train_samples} samples in {len(train_chunks)} chunks")
    print(f"  Validation: {val_samples} samples in {len(val_chunks) if val_chunks else 0} chunks")
    print(f"  Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())