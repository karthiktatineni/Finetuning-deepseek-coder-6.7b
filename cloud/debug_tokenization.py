#!/usr/bin/env python3
"""Debug script to test tokenization step by step"""

import os
import sys
import json
import yaml

print("=" * 60)
print("Tokenization Debug Script")
print("=" * 60)

# Step 1: Test config loading
print("\n1. Testing config loading...")
try:
    with open('config/cloud.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   Config keys: {list(config.keys())}")
    print(f"   Has 'model': {'model' in config}")
    print(f"   Has 'dataset': {'dataset' in config}")
    print(f"   Has 'io': {'io' in config}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Step 2: Test dataset loading (check file size first)
print("\n2. Checking dataset files...")
train_path = './datasets/merged/train.json'
val_path = './datasets/merged/validation.json'

if os.path.exists(train_path):
    file_size = os.path.getsize(train_path)
    print(f"   Train file size: {file_size / 1024 / 1024:.2f} MB")
else:
    print(f"   ERROR: Train file not found")
    sys.exit(1)

# Step 3: Try loading with line count
print("\n3. Testing minimal JSON load...")
try:
    import json
    # First try to just parse without loading all
    with open(train_path, 'r') as f:
        # Read just first 1000 chars to check structure
        preview = f.read(1000)
        print(f"   File preview: {preview[:200]}...")
        
    # Now try actual load with limit
    print("   Attempting JSON load...")
    with open(train_path, 'r') as f:
        import ijson
        # Stream through file to count items
        count = 0
        for item in ijson.items(f, 'item'):
            count += 1
            if count >= 10:  # Just check first 10
                break
        print(f"   Sample load successful. File appears to be valid JSON.")
except ImportError:
    print("   ijson not available, trying regular json.load...")
    try:
        with open(train_path, 'r') as f:
            data = json.load(f)
        print(f"   Loaded {len(data)} samples")
    except Exception as e:
        print(f"   ERROR: {e}")
        sys.exit(1)
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("\n✓ Basic tests passed. The issue may be in later steps.")
print("  The tokenization process might be failing during:")
print("  - HuggingFace Dataset creation")
print("  - Tokenizer processing")
print("  - Dataset mapping")
print("\nSuggestion: Run with 1 sample to isolate the issue.")