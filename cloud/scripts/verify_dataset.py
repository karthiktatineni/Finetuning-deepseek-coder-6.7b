#!/usr/bin/env python3
"""
Verify dataset integrity and compatibility with training pipeline.
Checks data quality, schema, and format requirements.
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


def check_required_columns(df, instruction_field, response_field):
    """Check if required columns exist."""
    required = [instruction_field, response_field]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"✗ Missing required columns: {missing}")
        return False
    else:
        print(f"✓ Required columns found: {required}")
        return True


def check_data_types(df, instruction_field, response_field):
    """Check if columns have correct data types."""
    issues = []
    
    for col in [instruction_field, response_field]:
        if col in df.columns:
            if df[col].dtype != 'object':
                issues.append(f" {col} has dtype {df[col].dtype}, should be string/text")
    
    if issues:
        print(f"✗ Data type issues:{''.join(issues)}")
        return False
    else:
        print(f"✓ Data types are correct")
        return True


def check_null_values(df, instruction_field, response_field):
    """Check for null values in critical columns."""
    issues = []
    
    for col in [instruction_field, response_field]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100
            if null_count > 0:
                issues.append(f"{col}: {null_count} null values ({null_percent:.1f}%)")
    
    if issues:
        print(f"⚠ Null values found: {', '.join(issues)}")
        return False
    else:
        print(f"✓ No null values in critical columns")
        return True


def check_duplicates(df):
    """Check for duplicate rows."""
    duplicate_count = df.duplicated().sum()
    duplicate_percent = (duplicate_count / len(df)) * 100
    
    if duplicate_count > 0:
        print(f"⚠ Duplicates found: {duplicate_count} ({duplicate_percent:.1f}%)")
        return False
    else:
        print(f"✓ No duplicate rows")
        return True


def check_sample_quality(df, instruction_field, response_field, sample_size=10):
    """Check quality of sample data."""
    print(f"\nChecking sample quality ({sample_size} samples)...")
    
    issues = []
    
    samples = df.head(sample_size).to_dict('records')
    
    for i, sample in enumerate(samples[:min(sample_size, len(samples))]):
        instruction = str(sample.get(instruction_field, ""))
        response = str(sample.get(response_field, ""))
        
        if not instruction.strip():
            issues.append(f"Sample {i+1}: Empty instruction")
        if not response.strip():
            issues.append(f"Sample {i+1}: Empty response")
        if len(instruction) < 10:
            issues.append(f"Sample {i+1}: Very short instruction ({len(instruction)} chars)")
        if len(response) < 10:
            issues.append(f"Sample {i+1}: Very short response ({len(response)} chars)")
    
    if issues:
        print(f"⚠ Quality issues: {', '.join(issues)}")
        return False
    else:
        print(f"✓ Sample quality looks good")
        return True


def check_text_length_distribution(df, instruction_field, response_field):
    """Check text length distribution."""
    print(f"\nText length analysis:")
    
    for col in [instruction_field, response_field]:
        if col in df.columns:
            lengths = df[col].dropna().apply(len)
            print(f"{col}:")
            print(f"  Mean: {lengths.mean():.1f} chars")
            print(f"  Median: {lengths.median():.1f} chars")
            print(f"  Max: {lengths.max()} chars")
            print(f"  Min: {lengths.min()} chars")
            
            # Warn about very long texts
            very_long = lengths > 4000
            if very_long.sum() > 0:
                print(f"  ⚠ Very long texts (>4000 chars): {very_long.sum()} ({very_long.sum()/len(df)*100:.1f}%)")


def check_encoding_issues(df, instruction_field, response_field):
    """Check for potential encoding issues."""
    print(f"\nEncoding check:")
    
    issues = []
    
    for col in [instruction_field, response_field]:
        if col in df.columns:
            for idx, value in df[col].dropna().head(100).items():
                try:
                    str(value).encode('utf-8')
                except UnicodeEncodeError as e:
                    issues.append(f"{col} row {idx}: Encoding error - {e}")
                    break
    
    if issues:
        print(f"⚠ Encoding issues: {', '.join(issues)}")
        return False
    else:
        print(f"✓ No encoding issues detected")
        return True


def check_dataset_size(df):
    """Check if dataset has sufficient size."""
    size = len(df)
    
    if size < 100:
        print(f"✗ Dataset too small: {size} samples (minimum: 100)")
        return False
    elif size < 1000:
        print(f"⚠ Small dataset: {size} samples (recommended: 1000+)")
        return True
    else:
        print(f"✓ Dataset size: {size} samples")
        return True


def check_for_special_chars(df, instruction_field, response_field):
    """Check for problematic special characters."""
    print(f"\nSpecial character check:")
    
    issues = []
    
    for col in [instruction_field, response_field]:
        if col in df.columns:
            for idx, value in df[col].dropna().head(50).items():
                if isinstance(value, str):
                    # Check for control characters except newlines and tabs
                    if any(ord(c) < 32 and c not in ['\n', '\r', '\t'] for c in value):
                        issues.append(f"{col} row {idx}: Contains control characters")
                        break
    
    if issues:
        print(f"⚠ Special character issues: {', '.join(issues)}")
        return False
    else:
        print(f"✓ No problematic special characters")
        return True


def generate_summary_report(checks_passed, total_checks, dataset_name):
    """Generate verification summary."""
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION SUMMARY: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("✓ Dataset is ready for processing!")
        return True
    elif checks_passed >= total_checks * 0.7:
        print("⚠ Dataset has minor issues but can proceed")
        return True
    else:
        print("✗ Dataset has significant issues, requires fixing")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify dataset integrity and quality')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--instruction', type=str, required=True,
                        help='Instruction field name')
    parser.add_argument('--response', type=str, required=True,
                        help='Response field name')
    parser.add_argument('--strict', action='store_true',
                        help='Fail on any warnings (not just errors)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Verification Tool")
    print("=" * 60)
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"✗ Dataset file not found: {args.dataset}")
        return 1
    
    print(f"\nVerifying: {args.dataset}")
    print(f"Instruction field: {args.instruction}")
    print(f"Response field: {args.response}")
    
    df = load_dataset(args.dataset)
    if df is None:
        return 1
    
    # Run verification checks
    checks = []
    
    print(f"\n{'=' * 60}")
    print(f"Running verification checks")
    print(f"{'=' * 60}")
    
    checks.append(("Required columns", check_required_columns(df, args.instruction, args.response)))
    checks.append(("Data types", check_data_types(df, args.instruction, args.response)))
    checks.append(("Null values", check_null_values(df, args.instruction, args.response)))
    checks.append(("Duplicates", check_duplicates(df)))
    checks.append(("Dataset size", check_dataset_size(df)))
    checks.append(("Encoding", check_encoding_issues(df, args.instruction, args.response)))
    checks.append(("Special characters", check_for_special_chars(df, args.instruction, args.response)))
    checks.append(("Sample quality", check_sample_quality(df, args.instruction, args.response)))
    
    # Additional analysis
    check_text_length_distribution(df, args.instruction, args.response)
    
    # Generate summary
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    success = generate_summary_report(passed, total, dataset_path.name)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())