#!/usr/bin/env python3
"""
Simple preprocessing script for HuggingFace datasets.
Avoids circular import issues by using direct imports.
"""

import os
import sys
import json
from pathlib import Path

# Import datasets module directly
import datasets


def load_hf_dataset(dataset_path):
    """Load HuggingFace dataset from disk."""
    try:
        dataset = datasets.load_from_disk(dataset_path)
        print(f"✓ Loaded {len(dataset)} samples from {dataset_path}")
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def get_field_mapping(dataset_name, columns):
    """Get field mapping for different dataset types."""
    mappings = {
        "opencoder_stage1": ("instruction", "output"),
        "opencoder_stage2": ("instruction", "output"),
        "codealpaca": ("instruction", "output"),
        "apps": ("question", "solutions"),
        "codesearchnet": ("func_documentation_string", "code"),
        "classeval": ("instruction", "code"),
        "python_alpaca": ("instruction", "output"),
        "code_alpaca_20k": ("instruction", "output"),
    }
    
    if dataset_name in mappings:
        return mappings[dataset_name]
    
    # Auto-detect
    possible_instruction = ["instruction", "question", "prompt", "input", "func_documentation_string"]
    possible_response = ["response", "output", "answer", "code", "solutions"]
    
    instruction_field = next((col for col in possible_instruction if col in columns), columns[0])
    response_field = next((col for col in possible_response if col in columns), columns[1] if len(columns) > 1 else columns[0])
    
    return instruction_field, response_field


def process_dataset(dataset, dataset_name, output_file, max_samples=None):
    """Process dataset into chat format with streaming to avoid OOM."""
    columns = list(dataset.features.keys())
    print(f"Columns: {columns}")
    
    instruction_field, response_field = get_field_mapping(dataset_name, columns)
    print(f"Using fields: {instruction_field} -> {response_field}")
    
    processed_count = 0
    failed_count = 0
    
    # Stream processing to avoid OOM with large datasets
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            if max_samples and processed_count >= max_samples:
                break
                
            try:
                instruction = item.get(instruction_field, "")
                response = item.get(response_field, "")
                
                # Handle list responses
                if isinstance(response, list):
                    response = "\n".join(str(r) for r in response)
                
                # Skip empty
                if not instruction or not response:
                    failed_count += 1
                    continue
                
                # Create chat format
                chat_item = {
                    "conversations": [
                        {"from": "human", "value": str(instruction)},
                        {"from": "gpt", "value": str(response)}
                    ]
                }
                
                # Write immediately to file instead of accumulating in memory
                f.write(json.dumps(chat_item, ensure_ascii=False) + '\n')
                processed_count += 1
                
                # Progress indicator for large datasets
                if processed_count % 10000 == 0:
                    print(f"  Processed {processed_count} samples...")
                    
            except Exception as e:
                failed_count += 1
                continue
    
    print(f"✓ Processed {processed_count} samples (failed: {failed_count})")
    return processed_count


def save_processed_data(data, output_path):
    """Save processed data to JSON."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving: {e}")
        return False


def convert_jsonl_to_json(input_jsonl, output_json):
    """Convert JSONL format to JSON array format."""
    try:
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = [json.loads(line.strip()) for line in lines if line.strip()]
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Converted {len(data)} samples from JSONL to JSON")
        return True
    except Exception as e:
        print(f"✗ Error converting: {e}")
        return False


def main():
    # Parse arguments - handle --config flag for compatibility
    config = None
    input_path = None
    dataset_name = None
    output_dir = "./datasets/processed"
    max_samples = None
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config = args[i + 1]
            i += 2
        elif args[i] == "--input" and i + 1 < len(args):
            input_path = args[i + 1]
            i += 2
        elif args[i] == "--max-samples" and i + 1 < len(args):
            max_samples = int(args[i + 1])
            i += 2
        else:
            # positional arguments
            if input_path is None:
                input_path = args[i]
            elif dataset_name is None:
                dataset_name = args[i]
            else:
                output_dir = args[i]
            i += 1
    
    if not input_path:
        print("Usage: python preprocess_simple.py [--config config.yaml] --input <input_path> [--max-samples N] [dataset_name] [output_dir]")
        return 1
    
    print("=" * 60)
    print("Simple Dataset Preprocessing")
    print("=" * 60)
    print(f"Input: {input_path}")
    
    if max_samples:
        print(f"Max samples: {max_samples}")
    
    if not dataset_name:
        dataset_name = Path(input_path).name
    print(f"Dataset: {dataset_name}")
    
    # Setup output paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use JSONL for streaming (memory efficient)
    jsonl_file = output_path / f"{dataset_name}_processed.jsonl"
    final_json_file = output_path / f"{dataset_name}_processed.json"
    
    print(f"Output: {final_json_file}")
    
    # Load dataset
    dataset = load_hf_dataset(input_path)
    if dataset is None:
        return 1
    
    # Process with streaming
    processed_count = process_dataset(dataset, dataset_name, str(jsonl_file), max_samples)
    if processed_count == 0:
        return 1
    
    # Convert JSONL to JSON array format for compatibility
    if convert_jsonl_to_json(str(jsonl_file), str(final_json_file)):
        # Clean up JSONL file
        jsonl_file.unlink()
        print(f"\n✓ Complete! Output: {final_json_file}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())