#!/usr/bin/env python3
"""
Inspect datasets to understand schema and data structure.
Useful for preprocessing configuration.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
from collections import Counter


def load_dataset(filepath):
    """Load dataset based on file type."""
    try:
        if filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        elif filepath.endswith('.jsonl'):
            return pd.read_json(filepath, lines=True)
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            print(f"✗ Unsupported file type: {filepath}")
            return None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def inspect_schema(df, dataset_name):
    """Inspect dataset schema and statistics."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\nColumns:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_percent = (null_count / len(df)) * 100
        print(f"  {col:30} {dtype:15} {null_count:6} null ({null_percent:5.1f}%)")
    
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return df


def show_sample_data(df, num_samples=3):
    """Show sample data rows."""
    if num_samples > len(df):
        num_samples = len(df)
    
    print(f"\nSample Data ({num_samples} rows):")
    print(f"{'=' * 60}")
    
    for idx in range(num_samples):
        print(f"\n--- Row {idx + 1} ---")
        for col in df.columns:
            value = df.iloc[idx][col]
            if pd.isnull(value):
                value_str = "null"
            elif isinstance(value, str):
                value_str = value[:100] + ("..." if len(value) > 100 else "")
            else:
                value_str = str(value)[:100]
            print(f"{col:30}: {value_str}")


def analyze_text_columns(df, max_length_analysis=True):
    """Analyze text columns for length and content."""
    print(f"\nText Column Analysis:")
    print(f"{'=' * 60}")
    
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        if df[col].dtype == 'object':
            # Sample non-null values
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_length = non_null.apply(len).mean()
                max_length = non_null.apply(len).max()
                min_length = non_null.apply(len).min()
                
                print(f"\n{col}:")
                print(f"  Average length: {avg_length:.1f} chars")
                print(f"  Max length: {max_length} chars")
                print(f"  Min length: {min_length} chars")
                
                if max_length_analysis:
                    long_texts = non_null[non_null.apply(len) > 2000]
                    if len(long_texts) > 0:
                        print(f"  Samples > 2000 chars: {len(long_texts)} ({len(long_texts)/len(non_null)*100:.1f}%)")


def detect_instruction_format(df):
    """Detect if dataset follows instruction/response format."""
    print(f"\nInstruction Format Detection:")
    print(f"{'=' * 60}")
    
    # Common instruction field names
    instruction_candidates = ['instruction', 'prompt', 'question', 'input', 'user_message']
    response_candidates = ['response', 'output', 'answer', 'assistant_message', 'solution']
    
    detected_instruction = None
    detected_response = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(candidate in col_lower for candidate in instruction_candidates):
            detected_instruction = col
        if any(candidate in col_lower for candidate in response_candidates):
            detected_response = col
    
    if detected_instruction and detected_response:
        print(f"✓ Detected instruction format:")
        print(f"  Instruction field: {detected_instruction}")
        print(f"  Response field: {detected_response}")
    elif detected_instruction or detected_response:
        partial = detected_instruction or detected_response
        print(f"⚠ Partial format detection: {partial}")
    else:
        print("⚠ No standard instruction format detected")
    
    return detected_instruction, detected_response


def suggest_preprocessing_config(df, instruction_field=None, response_field=None):
    """Suggest preprocessing configuration based on analysis."""
    print(f"\nPreprocessing Configuration Suggestion:")
    print(f"{'=' * 60}")
    
    if instruction_field and response_field:
        print(f"instruction_field: \"{instruction_field}\"")
        print(f"response_field: \"{response_field}\"")
    else:
        print("instruction_field: <MANUALLY_SPECIFY>")
        print(f"response_field: <MANUALLY_SPECIFY>")
    
    # Suggest sequence length
    all_text_columns = df.select_dtypes(include=['object']).columns
    total_text = ""
    for col in all_text_columns:
        samples = df[col].dropna().head(10).astype(str)
        total_text += " ".join(samples.values)
    
    avg_token_length = len(total_text) / len(total_text.split())
    estimated_max_tokens = int(max(df[all_text_columns].applymap(str).apply(lambda x: len(str(x))).max()))
    
    suggested_max_length = min(4096, estimated_max_tokens + 512)
    print(f"max_seq_length: {suggested_max_length}")
    
    # Suggest preprocessing options
    print(f"\nRecommended preprocessing:")
    print(f"remove_duplicates: true  (Check duplicates: {df.duplicated().sum()})")
    print(f"normalize_text: true")
    print(f"shuffle: true")


def main():
    parser = argparse.ArgumentParser(description='Inspect datasets to understand structure')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of sample rows to display')
    parser.add_argument('--analyze-text', action='store_true',
                        help='Analyze text columns in detail')
    parser.add_argument('--suggest-config', action='store_true',
                        help='Suggest preprocessing configuration')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Inspection Tool")
    print("=" * 60)
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"✗ Dataset file not found: {args.dataset}")
        return 1
    
    print(f"\nLoading dataset: {args.dataset}")
    df = load_dataset(args.dataset)
    
    if df is None:
        return 1
    
    # Inspect schema
    df = inspect_schema(df, dataset_path.name)
    
    # Show sample data
    show_sample_data(df, args.samples)
    
    # Analyze text columns if requested
    if args.analyze_text:
        analyze_text_columns(df)
    
    # Detect instruction format and suggest config if requested
    if args.suggest_config:
        instruction_field, response_field = detect_instruction_format(df)
        if instruction_field and response_field:
            suggest_preprocessing_config(df, instruction_field, response_field)
        else:
            print("\n⚠ Could not auto-detect instruction format")
            suggest_preprocessing_config(df)
    
    print(f"\n{'=' * 60}")
    print("Inspection complete!")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())