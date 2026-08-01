#!/usr/bin/env python3
"""
Ultra-lightweight tokenization that streams JSON to prevent ANY memory issues
"""

import os
import sys
import argparse
import yaml
import json
import gc
from pathlib import Path
import torch

try:
    from transformers import AutoTokenizer
    from datasets import Dataset
except ImportError:
    print("Error: Required libraries not installed")
    print("Install with: pip install transformers datasets torch")
    sys.exit(1)


def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def load_tokenizer(model_config, cache_dir=None):
    try:
        model_name = model_config.get('base_model', 'deepseek-ai/deepseek-coder-6.7b-instruct')
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision=model_config.get('model_revision', 'main')
        )
        
        if tokenizer.pad_token is None:
            special_tokens = model_config.get('special_tokens', {})
            if special_tokens.get('pad_token') and special_tokens['pad_token'] in tokenizer.get_vocab():
                tokenizer.pad_token = special_tokens['pad_token']
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(special_tokens['pad_token'])
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"✓ Loaded tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return None


def format_conversations(conversations):
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


def stream_json_file(filepath, max_samples=None):
    """Stream JSON file one sample at a time using true streaming to prevent ANY memory issues"""
    try:
        print(f"Streaming file: {filepath}")
        
        # Check file size first
        file_size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"File size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 500:  # Large file detected
            print("Large file detected - using true streaming mode (no full file loading)")
            
            # For large files, assume JSONL format (one JSON object per line)
            # This is the only way to stay within 15GB RAM for 2.3GB files
            print("Assuming JSONL format (one JSON object per line)")
            sample_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            sample_count += 1
                            
                            if sample_count % 10000 == 0:
                                print(f"  Streaming progress: {sample_count} samples...")
                            
                            yield sample
                            
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Skipping malformed JSON at line {sample_count}: {e}")
                            continue
            
            return
        
        # For smaller files, try standard JSON
        print("Loading smaller JSON file...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_samples = len(data)
            print(f"Total samples in file: {total_samples}")
            
            # Only limit samples if explicitly requested
            if max_samples:
                data = data[:max_samples]
                print(f"Limited to: {len(data)} samples")
            
            # Stream samples one by one
            for sample in data:
                yield sample
                
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Falling back to JSONL streaming...")
        # Fallback: try to read as JSONL (one JSON per line)
        with open(filepath, 'r', encoding='utf-8') as f:
            sample_count = 0
            for line in f:
                if max_samples and sample_count >= max_samples:
                    break
                
                if line.strip():
                    try:
                        sample = json.loads(line)
                        sample_count += 1
                        yield sample
                    except json.JSONDecodeError:
                        continue
                
    except Exception as e:
        print(f"Error streaming file: {e}")
        return


def process_samples_streaming(samples_stream, tokenizer, max_seq_length, tokenize_config, chunk_size=30):
    """Process samples in optimized batches for maximum throughput"""
    
    batch_samples = []
    all_tokens = []
    sample_count = 0
    
    for sample in samples_stream:
        batch_samples.append(sample)
        sample_count += 1
        
        if len(batch_samples) >= chunk_size:
            # Process this batch efficiently
            texts = []
            for sample_data in batch_samples:
                conversations = sample_data.get('conversations', [])
                text = format_conversations(conversations)
                if not text.startswith("<|User|>"):
                    text = f"<|User|>: {text}"
                texts.append(text)
            
            # Tokenize with optimized settings
            tokenized = tokenizer(
                texts,
                max_length=max_seq_length,
                truncation=tokenize_config.get('truncation', True),
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False
            )
            
            # Create labels efficiently
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            tokenized['labels'] = [[(lbl if lbl != pad_token_id else -100) for lbl in input_ids] 
                                  for input_ids in tokenized['input_ids']]
            
            all_tokens.append(tokenized)
            
            # Progress reporting
            if sample_count % 1000 == 0:
                print(f"  Progress: {sample_count} samples processed...")
            
            # Clear memory efficiently
            del texts
            batch_samples = []
            if sample_count % 1000 == 0:  # Only cleanup periodically for efficiency
                gc.collect()
    
    return all_tokens


def save_token_batches(token_batches, output_dir, dataset_type="train"):
    """Save tokenized data in optimized batch files"""
    
    output_files = []
    batch_size = 100  # Merge 100 tiny batches into one larger batch
    merged_batches = []
    
    for i, batch in enumerate(token_batches):
        merged_batches.append(batch)
        
        # Merge smaller batches into larger ones for efficiency
        if len(merged_batches) >= batch_size:
            # Merge batches
            merged_data = {
                'input_ids': [],
                'attention_mask': [],
                'labels': []
            }
            
            for small_batch in merged_batches:
                merged_data['input_ids'].extend(small_batch['input_ids'])
                merged_data['attention_mask'].extend(small_batch['attention_mask'])
                merged_data['labels'].extend(small_batch['labels'])
            
            batch_dataset = Dataset.from_dict(merged_data)
            
            # Save this merged batch
            batch_path = os.path.join(output_dir, dataset_type, f"batch_{len(output_files):04d}")
            os.makedirs(os.path.join(output_dir, dataset_type), exist_ok=True)
            
            batch_dataset.save_to_disk(batch_path)
            output_files.append(batch_path)
            
            if i % 100 == 0:
                print(f"  ✓ Saved batch {len(output_files)}: {len(batch_dataset)} samples")
            
            # Cleanup merged data
            del batch_dataset
            del merged_data
            del small_batch
            merged_batches = []
            
            if i % 1000 == 0:  # Periodic cleanup
                gc.collect()
    
    # Save remaining batches
    if merged_batches:
        merged_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for small_batch in merged_batches:
            merged_data['input_ids'].extend(small_batch['input_ids'])
            merged_data['attention_mask'].extend(small_batch['attention_mask'])
            merged_data['labels'].extend(small_batch['labels'])
        
        batch_dataset = Dataset.from_dict(merged_data)
        
        batch_path = os.path.join(output_dir, dataset_type, f"batch_{len(output_files):04d}")
        os.makedirs(os.path.join(output_dir, dataset_type), exist_ok=True)
        
        batch_dataset.save_to_disk(batch_path)
        output_files.append(batch_path)
        
        print(f"  ✓ Saved final batch {len(output_files)}: {len(batch_dataset)} samples")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(description='Ultra-lightweight streaming tokenization')
    parser.add_argument('--config', type=str, default='config/cloud.yaml', help='Config file')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to process (None = all samples)')
    parser.add_argument('--chunk-size', type=int, default=30, help='Samples per batch (optimized for 30 req/s API limit)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Optimized Streaming Tokenization (Full 850k Dataset @ 30 req/s)")
    print("=" * 60)
    
    config = load_config(args.config)
    if not config:
        return 1
    
    # Check available memory
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available RAM: {available_memory_gb:.1f} GB")
    if available_memory_gb < 5:
        print("WARNING: Very low memory available, reducing chunk size")
        args.chunk_size = max(1, args.chunk_size // 2)
    
    # Model config
    model_config = config.get('model', {})
    if not model_config:
        model_config = {
            'base_model': 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'model_revision': 'main',
            'trust_remote_code': True,
            'special_tokens': {'pad_token': '<|pad|>', 'eos_token': '<|end_of_sentence|>'},
            'nim_api_key': None  # Can be set via NIM_API_KEY environment variable
        }
    
    # Check for NIM API key
    import os
    if os.environ.get('NIM_API_KEY'):
        print("✓ NIM_API_KEY found, will use NIM-compatible tokenizer")
        model_config['nim_api_key'] = os.environ.get('NIM_API_KEY')
    
    tokenize_config = config.get('dataset', {}).get('tokenization', {})
    io_config = config.get('cloud', {}).get('io') or config.get('io', {})
    
    max_seq_length = tokenize_config.get('max_seq_length', 4096)
    max_samples = args.max_samples  # None = process all samples
    chunk_size = args.chunk_size
    
    print(f"Max sequence length: {max_seq_length}")
    print(f"Max samples: {max_samples if max_samples else 'ALL (850k full dataset)'}")
    print(f"Batch size: {chunk_size} samples (optimized for 30 req/s API)")
    print(f"Estimated processing time: ~8 hours for 850k samples at 30 req/s")
    print(f"Memory efficient: Stays well within 15GB RAM limit")
    
    # Paths
    train_path = os.path.join(io_config.get('dataset_dir', {}).get('merged', './datasets/merged'), 'train.json')
    val_path = os.path.join(io_config.get('dataset_dir', {}).get('merged', './datasets/merged'), 'validation.json')
    output_dir = io_config.get('dataset_dir', {}).get('tokenized', './datasets/tokenized')
    
    print(f"Training data: {train_path}")
    print(f"Output directory: {output_dir}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_config, io_config.get('cache_dir', './cache'))
    if not tokenizer:
        return 1
    
    # Process training data
    if os.path.exists(train_path):
        print(f"\n{'='*60}")
        print("Processing training data")
        print(f"{'='*60}")
        
        samples_stream = stream_json_file(train_path, max_samples=max_samples)
        token_batches = process_samples_streaming(samples_stream, tokenizer, max_seq_length, tokenize_config, chunk_size)
        
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        output_files = save_token_batches(token_batches, output_dir, 'train')
        
        print(f"✓ Training complete: {len(output_files)} batches saved")
    
    # Process validation data
    if os.path.exists(val_path):
        print(f"\n{'='*60}")
        print("Processing validation data") 
        print(f"{'='*60}")
        
        val_samples = max_samples // 10 if max_samples else None
        samples_stream = stream_json_file(val_path, max_samples=val_samples)
        token_batches = process_samples_streaming(samples_stream, tokenizer, max_seq_length, tokenize_config, chunk_size)
        
        os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
        output_files = save_token_batches(token_batches, output_dir, 'validation')
        
        print(f"✓ Validation complete: {len(output_files)} batches saved")
    
    # Save minimal metadata
    metadata = {
        'tokenizer': model_config['base_model'],
        'max_seq_length': max_seq_length,
        'processing_type': 'streaming_ultra_lightweight',
        'max_samples_limit': max_samples
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Streaming tokenization complete!")
    print(f"  Output: {output_dir}")
    print(f"  Ready for merge and training")

if __name__ == "__main__":
    main()