#!/usr/bin/env python3
"""
Merge multiple preprocessed datasets into a single training dataset.
Supports various merging strategies: concatenation, sampling, weighted sampling.
"""

import os
import sys
import argparse
import yaml
import json
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm


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


def count_json_lines(filepath):
    """Count lines/samples in JSON file without loading full content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data)
    except Exception as e:
        print(f"Error counting lines in {filepath}: {e}")
        return 0


def load_processed_dataset(filepath, max_samples=None):
    """Load processed dataset with optional sample limit."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples and len(data) > max_samples:
            print(f"File has {len(data)} samples, limiting to {max_samples}")
            data = data[:max_samples]
        
        print(f"Loaded {len(data)} samples from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading dataset {filepath}: {e}")
        return None


def find_processed_datasets(processed_dir):
    """Find all processed datasets in the directory."""
    try:
        datasets = []
        
        if not os.path.exists(processed_dir):
            print(f"Processed directory not found: {processed_dir}")
            return []
        
        for file in os.listdir(processed_dir):
            if file.endswith('_processed.json'):
                filepath = os.path.join(processed_dir, file)
                datasets.append({
                    'name': Path(file).stem.replace('_processed', ''),
                    'path': filepath
                })
        
        print(f"Found {len(datasets)} processed datasets")
        return datasets
    except Exception as e:
        print(f"Error finding processed datasets: {e}")
        return []


def concatenate_datasets(datasets, max_samples=None):
    """Simple concatenation of all datasets with memory limits."""
    print(f"Concatenating {len(datasets)} datasets...")
    
    merged_data = []
    
    # Load datasets one at a time with limits to prevent OOM
    if max_samples:
        samples_per_dataset = max_samples // len(datasets)
        print(f"Limiting to {max_samples} total samples (~{samples_per_dataset} per dataset)")
    else:
        samples_per_dataset = None
    
    for i, dataset in enumerate(datasets):
        print(f"Processing {dataset['name']} ({i+1}/{len(datasets)})...")
        
        if samples_per_dataset:
            print(f"  Loading up to {samples_per_dataset} samples")
        
        # Load dataset with sample limit
        data = load_processed_dataset(dataset['path'], samples_per_dataset)
        
        if data:
            merged_data.extend(data)
            print(f"  Added {len(data)} samples from {dataset['name']}")
    
    print(f"Total samples after concatenation: {len(merged_data)}")
    
    return merged_data


def sample_datasets(datasets, samples_per_dataset=None, total_samples=None):
    """Sample from each dataset."""
    print(f"Sampling from {len(datasets)} datasets...")
    
    merged_data = []
    
    for dataset in tqdm(datasets, desc="Sampling datasets"):
        data = load_processed_dataset(dataset['path'])
        if not data:
            continue
        
        # Determine sample count
        if samples_per_dataset:
            sample_count = min(samples_per_dataset, len(data))
        elif total_samples:
            sample_count = min(total_samples // len(datasets), len(data))
        else:
            sample_count = len(data)
        
        # Sample from dataset
        sampled = random.sample(data, sample_count)
        merged_data.extend(sampled)
        print(f"  Sampled {sample_count} from {dataset['name']} (total: {len(data)})")
    
    print(f"Total samples after sampling: {len(merged_data)}")
    return merged_data


def weighted_sample_datasets(datasets, weights_dict=None, total_samples=None):
    """Sample from datasets with specified weights."""
    print(f"Weighted sampling from {len(datasets)} datasets...")
    
    dataset_data = []
    
    for dataset in datasets:
        data = load_processed_dataset(dataset['path'])
        if data:
            dataset_data.append({
                'name': dataset['name'],
                'data': data
            })
    
    if not dataset_data:
        print("No valid datasets to sample from")
        return []
    
    # Get weights
    if weights_dict:
        weights = [
            weights_dict.get(dataset['name'], weights_dict.get(dataset['name'].replace('-', '_'), 1.0))
            for dataset in dataset_data
        ]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
    else:
        probabilities = [1.0 / len(dataset_data)] * len(dataset_data)
    
    print(f"Dataset weights and probabilities:")
    for dataset, prob in zip(dataset_data, probabilities):
        weight = weights_dict.get(dataset['name'], 1.0) if weights_dict else 1.0
        print(f"  {dataset['name']}: weight={weight:.2f}, prob={prob:.3f}")
    
    # Determine total samples
    if total_samples:
        target_samples = total_samples
    else:
        target_samples = sum(len(dataset['data']) for dataset in dataset_data)
    
    print(f"Target total samples: {target_samples}")
    
    # Calculate samples per dataset based on weights
    merged_data = []
    for dataset, prob in zip(dataset_data, probabilities):
        sample_count = int(target_samples * prob)
        sample_count = min(sample_count, len(dataset['data']))
        
        sampled = random.sample(dataset['data'], sample_count)
        merged_data.extend(sampled)
        print(f"  Sampled {sample_count} from {dataset['name']} ({prob*100:.1f}%)")
    
    print(f"Total samples after weighted sampling: {len(merged_data)}")
    return merged_data


def shuffle_dataset(data, seed=42):
    """Shuffle dataset with reproducible seed."""
    print(f"Shuffling dataset with seed {seed}...")
    
    random.seed(seed)
    random.shuffle(data)
    
    return data


def split_dataset(data, train_split=0.95, seed=42):
    """Split dataset into train and validation sets."""
    print(f"Splitting dataset: {train_split*100:.1f}% train")
    
    total_samples = len(data)
    train_count = int(total_samples * train_split)
    
    # Split
    train_data = data[:train_count]
    val_data = data[train_count:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data


def save_dataset(data, output_path, dataset_type="merged"):
    """Save dataset to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {dataset_type} dataset ({len(data)} samples) to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False


def analyze_merged_dataset(data):
    """Analyze merged dataset statistics."""
    print(f"\nMerged dataset analysis:")
    print(f"  Total samples: {len(data)}")
    
    if len(data) == 0:
        return
    
    # Analyze conversation lengths
    conv_lengths = []
    total_chars = 0
    
    for item in data[:1000]:  # Sample for performance
        if 'conversations' in item and isinstance(item['conversations'], list):
            total_text = ""
            for msg in item['conversations']:
                value = msg.get('value', '')
                if isinstance(value, str):
                    total_text += value
            conv_lengths.append(len(total_text))
            total_chars += len(total_text)
    
    if conv_lengths:
        avg_length = sum(conv_lengths) / len(conv_lengths)
        max_length = max(conv_lengths)
        min_length = min(conv_lengths)
        
        print(f"  Avg conversation length: {avg_length:.0f} chars")
        print(f"  Max conversation length: {max_length} chars")
        print(f"  Min conversation length: {min_length} chars")
    
    # Total characters
    if total_chars > 0:
        total_gb = total_chars / 1024 / 1024 / 1024
        print(f"  Total text size: ~{total_gb:.2f} GB (estimated)")


def main():
    parser = argparse.ArgumentParser(description='Merge multiple datasets into one')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory with processed datasets')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for merged dataset')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['concat', 'sample', 'weighted'],
                        help='Merging strategy (overrides config)')
    parser.add_argument('--samples-per-dataset', type=int, default=None,
                        help='Samples per dataset (for sample strategy)')
    parser.add_argument('--total-samples', type=int, default=None,
                        help='Total samples (for sample/weighted strategies)')
    parser.add_argument('--train-split', type=float, default=None,
                        help='Train/validation split ratio')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Merging")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get dataset configuration
    dataset_config = config.get('dataset', {})
    merging_config = dataset_config.get('merging', {})
    
    # Get override values
    strategy = args.strategy or merging_config.get('strategy', 'concat')
    train_split = args.train_split or merging_config.get('train_split', 0.95)
    seed = args.seed or merging_config.get('seed', 42)
    
    # Determine directories
    io_config = config.get('io', {})
    
    # fallback to cloud.io if io is empty
    if not io_config:
        io_config = config.get('cloud', {}).get('io', {})
    
    if args.input_dir:
        input_dir = args.input_dir
    elif io_config and io_config.get('dataset_dir', {}).get('processed'):
        input_dir = io_config['dataset_dir']['processed']
    else:
        input_dir = './datasets/processed'
    
    if args.output_dir:
        output_dir = args.output_dir
    elif io_config and io_config.get('dataset_dir', {}).get('merged'):
        output_dir = io_config['dataset_dir']['merged']
    else:
        output_dir = './datasets/merged'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Merging strategy: {strategy}")
    print(f"Train split: {train_split:.2f}")
    print(f"Random seed: {seed}")
    
    # Find processed datasets
    datasets = find_processed_datasets(input_dir)
    if not datasets:
        print("No processed datasets found to merge")
        return 1
    
    # Merge datasets
    merged_data = []
    
    if strategy == 'concat':
        max_samples = merging_config.get('max_samples', 100000)  # Default limit to prevent OOM
        merged_data = concatenate_datasets(datasets, max_samples)
    elif strategy == 'sample':
        samples_per_dataset = args.samples_per_dataset or merging_config.get('samples_per_dataset', 50000)
        total_samples = args.total_samples
        merged_data = sample_datasets(datasets, samples_per_dataset, total_samples)
    elif strategy == 'weighted':
        weights_dict = merging_config.get('weights', {})
        total_samples = args.total_samples
        merged_data = weighted_sample_datasets(datasets, weights_dict, total_samples)
    else:
        print(f"Unknown strategy: {strategy}")
        return 1
    
    if not merged_data:
        print("✗ Failed to merge datasets")
        return 1
    
    # Shuffle
    merged_data = shuffle_dataset(merged_data, seed)
    
    # Analyze merged dataset
    analyze_merged_dataset(merged_data)
    
    # Split into train/val
    train_data, val_data = split_dataset(merged_data, train_split, seed)
    
    # Save datasets
    train_output = os.path.join(output_dir, 'train.json')
    val_output = os.path.join(output_dir, 'validation.json')
    merged_output = os.path.join(output_dir, 'merged.json')
    
    if not save_dataset(merged_data, merged_output, "merged"):
        return 1
    
    if not save_dataset(train_data, train_output, "train"):
        return 1
    
    if not save_dataset(val_data, val_output, "validation"):
        return 1
    
    print(f"\n✓ Dataset merging complete!")
    print(f"  Train: {train_output} ({len(train_data)} samples)")
    print(f"  Validation: {val_output} ({len(val_data)} samples)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())