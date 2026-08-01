#!/usr/bin/env python3
"""
Preprocess individual datasets into standardized chat format.
Supports both HuggingFace Dataset format and direct file formats.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datasets import Dataset
import logging


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


def load_huggingface_dataset(dataset_path):
    """Load HuggingFace Dataset from disk."""
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"Loaded {len(dataset)} samples from HuggingFace format")
        return dataset
    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}")
        return None


def create_chat_format(instruction, response):
    """Create standardized chat format message."""
    return {
        "conversations": [
            {
                "from": "human",
                "value": instruction
            },
            {
                "from": "gpt", 
                "value": response
            }
        ]
    }


def process_huggingface_columns(dataset, dataset_name):
    """Convert HuggingFace dataset columns to chat format."""
    print(f"Processing {dataset_name}...")
    
    # Field mapping based on dataset structure
    field_mappings = {
        "opencoder_stage1": {"instruction": "instruction", "response": "response"},
        "opencoder_stage2": {"instruction": "instruction", "response": "response"},
        "codealpaca": {"instruction": "instruction", "response": "output"},
        "apps": {"instruction": "question", "response": "solutions"},
        "codesearchnet": {"instruction": "func_documentation_string", "response": "code"},
        "classeval": {"instruction": "instruction", "response": "code"}
    }
    
    # Get field mapping or try to auto-detect
    if dataset_name in field_mappings:
        instruction_field = field_mappings[dataset_name]["instruction"]
        response_field = field_mappings[dataset_name]["response"]
    else:
        # Auto-detect fields
        columns = list(dataset.features.keys())
        print(f"Available columns: {columns}")
        
        # Try to find instruction field
        possible_instruction_fields = ["instruction", "question", "prompt", "input", "func_documentation_string"]
        instruction_field = next((col for col in possible_instruction_fields if col in columns), columns[0])
        
        # Try to find response field
        possible_response_fields = ["response", "output", "answer", "code", "solutions"]
        response_field = next((col for col in possible_response_fields if col in columns), columns[1])
        
        print(f"Auto-detected fields: instruction={instruction_field}, response={response_field}")
    
    # Process dataset
    processed_data = []
    total_samples = len(dataset)
    
    for i, sample in enumerate(dataset):
        try:
            instruction = sample.get(instruction_field, "")
            response = sample.get(response_field, "")
            
            # Handle list responses (common in some datasets)
            if isinstance(response, list):
                response = "\n".join(str(item) for item in response)
            
            # Skip empty samples
            if not instruction or not response:
                continue
            
            # Create chat format
            chat_item = create_chat_format(str(instruction), str(response))
            processed_data.append(chat_item)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Processed {len(processed_data)}/{total_samples} samples")
    return processed_data


def save_processed_dataset(data, output_path):
    """Save processed dataset to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} processed samples to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving processed dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Preprocess HuggingFace datasets into standardized format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input HuggingFace dataset path or directory')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset name for field mapping')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for processed datasets')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HuggingFace Dataset Preprocessing")
    print("=" * 60)
    print(f"Input: {args.input}")
    
    # Determine dataset name from path if not provided
    if args.dataset_name is None:
        args.dataset_name = Path(args.input).name
    
    print(f"Dataset name: {args.dataset_name}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./datasets/processed"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.dataset_name}_processed.json")
    print(f"Output: {output_file}")
    
    # Load HuggingFace dataset
    dataset = load_huggingface_dataset(args.input)
    if dataset is None:
        return 1
    
    # Process dataset
    processed_data = process_huggingface_columns(dataset, args.dataset_name)
    if not processed_data:
        print("No data was processed successfully")
        return 1
    
    # Save processed dataset
    if save_processed_dataset(processed_data, output_file):
        print(f"\n✓ Preprocessing complete!")
        return 0
    else:
        print(f"\n✗ Failed to save processed dataset")
        return 1


if __name__ == "__main__":
    sys.exit(main())