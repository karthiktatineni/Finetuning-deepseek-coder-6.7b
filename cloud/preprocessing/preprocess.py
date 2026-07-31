#!/usr/bin/env python3
"""
Preprocess individual datasets into standardized chat format.
Supports instruction/response normalization and text cleaning.
"""

import os
import sys
import argparse
import yaml
import json
import re
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


def load_dataset(filepath):
    """Load dataset based on file type."""
    try:
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
            print(f"Loaded {len(df)} samples from Parquet file")
            return df
        elif filepath.endswith('.jsonl'):
            df = pd.read_json(filepath, lines=True)
            print(f"Loaded {len(df)} samples from JSONL file")
            return df
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)
                print(f"Loaded {len(df)} samples from JSON file")
                return df
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} samples from CSV file")
            return df
        else:
            print(f"Unsupported file type: {filepath}")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def normalize_text(text):
    """Normalize text content."""
    if text is None:
        return ""
    
    text = str(text).strip()
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text


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


def preprocess_dataset(df, dataset_config, processing_config):
    """Preprocess dataset according to configuration."""
    print(f"\nPreprocessing dataset...")
    
    # Get field mappings
    instruction_field = dataset_config.get('instruction_field', 'instruction')
    response_field = dataset_config.get('response_field', 'response')
    
    # Check if required fields exist
    missing = [f for f in [instruction_field, response_field] if f not in df.columns]
    if missing:
        print(f"Missing fields: {missing}")
        print(f"Available fields: {list(df.columns)}")
        return None
    
    print(f"Using fields: instruction={instruction_field}, response={response_field}")
    
    # Handle max_samples for testing
    max_samples = processing_config.get('max_samples', 0)
    if max_samples > 0 and max_samples < len(df):
        print(f"Warning: Processing only {max_samples} samples for testing")
        df = df.head(max_samples)
    
    # Normalize text if enabled
    normalize = processing_config.get('normalize_text', True)
    if normalize:
        print("Normalizing text content...")
        df[instruction_field] = df[instruction_field].apply(normalize_text)
        df[response_field] = df[response_field].apply(normalize_text)
    
    # Remove duplicates if enabled
    remove_duplicates = processing_config.get('remove_duplicates', True)
    if remove_duplicates:
        initial_count = len(df)
        df = df.drop_duplicates(subset=[instruction_field, response_field])
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate samples")
    
    # Remove rows with null values in critical fields
    initial_count = len(df)
    df = df.dropna(subset=[instruction_field, response_field])
    null_removed = initial_count - len(df)
    if null_removed > 0:
        print(f"Removed {null_removed} samples with null values")
    
    # Create chat format
    print("Creating chat format...")
    processed_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        instruction = str(row[instruction_field])
        response = str(row[response_field])
        
        # Skip empty samples
        if not instruction.strip() or not response.strip():
            continue
        
        # Create chat format
        chat_item = create_chat_format(instruction, response)
        processed_data.append(chat_item)
    
    processed_df = pd.DataFrame(processed_data)
    print(f"Preprocessed {len(processed_df)} samples")
    
    return processed_df


def save_processed_dataset(df, output_path):
    """Save processed dataset to JSON format."""
    try:
        # Convert to list of dicts
        data = df.to_dict('records')
        
        # Save as JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved processed dataset to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving processed dataset: {e}")
        return False


def inspect_dataset_structure(df):
    """Inspect and display dataset structure."""
    print(f"\nDataset structure:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Show first sample
    if len(df) > 0:
        print(f"\nFirst sample:")
        first_sample = df.iloc[0]
        for col in df.columns:
            value = first_sample[col]
            if isinstance(value, str):
                value = value[:200] + "..." if len(value) > 200 else value
            print(f"  {col}: {value}")


def validate_processed_dataset(df):
    """Validate processed dataset structure."""
    print(f"\nValidating processed dataset...")
    
    errors = []
    
    # Check required columns
    required_columns = ['conversations']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check conversation structure
    if 'conversations' in df.columns:
        for idx, conversations in enumerate(df['conversations'].head(10)):
            if not isinstance(conversations, list) or len(conversations) != 2:
                errors.append(f"Row {idx}: Invalid conversation structure")
                break
            
            first_msg = conversations[0]
            if not isinstance(first_msg, dict) or 'from' not in first_msg or 'value' not in first_msg:
                errors.append(f"Row {idx}: Invalid message format")
                break
            
            if first_msg['from'] != 'human':
                errors.append(f"Row {idx}: First message should be 'human'")
                break
    
    # Check for empty values
    empty_count = df['conversations'].apply(lambda x: len(x) == 0 or not any(msg.get('value') for msg in x)).sum()
    if empty_count > 0:
        errors.append(f"{empty_count} samples have empty conversations")
    
    if errors:
        print(f"✗ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✓ Processed dataset is valid")
        return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets into standardized format')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dataset file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output processed dataset file')
    parser.add_argument('--instruction-field', type=str, default=None,
                        help='Override instruction field name')
    parser.add_argument('--response-field', type=str, default=None,
                        help='Override response field name')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process (for testing)')
    parser.add_argument('--skip-normalization', action='store_true',
                        help='Skip text normalization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Preprocessing")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get processing configuration
    processing_config = config.get('dataset', {}).get('preprocessing', {})
    
    # Override configuration if provided
    if args.max_samples is not None:
        processing_config['max_samples'] = args.max_samples
    if args.skip_normalization:
        processing_config['normalize_text'] = False
    
    # Determine output directory
    io_config = config.get('cloud', {}).get('io')
    output_dir = io_config['dataset_dir']['processed']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine output file path
    if args.output:
        output_path = args.output
    else:
        input_file = Path(args.input)
        output_path = os.path.join(output_dir, f"{input_file.stem}_processed.json")
    
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    
    # Load dataset
    df = load_dataset(args.input)
    if df is None:
        return 1
    
    # Create dataset config
    dataset_config = {}
    if args.instruction_field:
        dataset_config['instruction_field'] = args.instruction_field
    if args.response_field:
        dataset_config['response_field'] = args.response_field
    
    # Process dataset
    processed_df = preprocess_dataset(df, dataset_config, processing_config)
    if processed_df is None:
        return 1
    
    # Inspect structure
    inspect_dataset_structure(processed_df)
    
    # Validate
    if not validate_processed_dataset(processed_df):
        print("✗ Validation failed, but saving anyway for inspection")
    
    # Save processed dataset
    if save_processed_dataset(processed_df, output_path):
        print(f"\n✓ Preprocessing complete!")
        return 0
    else:
        print(f"\n✗ Failed to save processed dataset")
        return 1


if __name__ == "__main__":
    sys.exit(main())