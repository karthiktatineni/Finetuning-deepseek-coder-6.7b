#!/usr/bin/env python3
"""
Merge tokenized chunks into single datasets for training
"""

import os
import json
from pathlib import Path
from datasets import Dataset, concatenate_datasets
import shutil

def merge_tokenized_chunks(chunk_dir, output_path):
    """Merge multiple tokenized dataset chunks into one."""
    try:
        print(f"Merging chunks from {chunk_dir}...")
        
        # Get all chunk directories
        chunk_dirs = sorted([d for d in Path(chunk_dir).iterdir() if d.is_dir() and d.name.startswith('chunk_')])
        
        if not chunk_dirs:
            print(f"No chunks found in {chunk_dir}")
            return None
        
        print(f"Found {len(chunk_dirs)} chunks")
        
        # Load and concatenate all chunks
        datasets = []
        for chunk_dir in chunk_dirs:
            try:
                chunk_dataset = Dataset.load_from_disk(str(chunk_dir))
                datasets.append(chunk_dataset)
                print(f"  Loaded chunk {chunk_dir.name}: {len(chunk_dataset)} samples")
            except Exception as e:
                print(f"  Error loading chunk {chunk_dir.name}: {e}")
        
        if not datasets:
            print("Failed to load any chunks")
            return None
        
        # Concatenate all datasets
        print("Concatenating datasets...")
        merged_dataset = concatenate_datasets(datasets)
        print(f"✓ Merged dataset: {len(merged_dataset)} samples")
        
        # Save merged dataset
        print(f"Saving to {output_path}...")
        merged_dataset.save_to_disk(output_path)
        print(f"✓ Saved merged dataset to {output_path}")
        
        return merged_dataset
        
    except Exception as e:
        print(f"Error merging chunks: {e}")
        return None

def main():
    tokenized_dir = './datasets/tokenized'
    
    print("=" * 60)
    print("Merging Tokenized Chunks")
    print("=" * 60)
    
    # Merge training chunks
    train_chunk_dir = os.path.join(tokenized_dir, 'train')
    train_output = os.path.join(tokenized_dir, 'train_merged')
    
    if os.path.exists(train_chunk_dir):
        train_dataset = merge_tokenized_chunks(train_chunk_dir, train_output)
    else:
        print(f"Training chunk directory not found: {train_chunk_dir}")
        train_dataset = None
    
    # Merge validation chunks
    val_chunk_dir = os.path.join(tokenized_dir, 'validation')
    val_output = os.path.join(tokenized_dir, 'validation_merged')
    
    if os.path.exists(val_chunk_dir):
        val_dataset = merge_tokenized_chunks(val_chunk_dir, val_output)
    else:
        print(f"Validation chunk directory not found: {val_chunk_dir}")
        val_dataset = None
    
    # Update metadata
    metadata_path = os.path.join(tokenized_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if train_dataset:
            metadata['train_samples'] = len(train_dataset)
            metadata['train_merged_path'] = train_output
        
        if val_dataset:
            metadata['val_samples'] = len(val_dataset)
            metadata['val_merged_path'] = val_output
        
        metadata['merged'] = True
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Updated metadata")
    
    print("\n✓ All chunks merged successfully!")
    if train_dataset:
        print(f"  Training: {len(train_dataset)} samples -> {train_output}")
    if val_dataset:
        print(f"  Validation: {len(val_dataset)} samples -> {val_output}")
    
    print(f"\nReady for training!")
    print(f"  python training/train.py --config config/cloud.yaml")

if __name__ == "__main__":
    main()