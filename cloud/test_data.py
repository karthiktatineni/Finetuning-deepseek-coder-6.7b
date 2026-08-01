#!/usr/bin/env python3
import json
import os
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Check dataset
train_path = 'datasets/merged/train.json'
val_path = 'datasets/merged/validation.json'

print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)

if os.path.exists(train_path):
    with open(train_path, 'r') as f:
        data = json.load(f)
    print(f"\n✓ Training file exists: {train_path}")
    print(f"  Total samples: {len(data)}")
    if len(data) > 0:
        print(f"  Sample structure:")
        print(json.dumps(data[0], indent=2))
        
        # Check role field
        if 'conversations' in data[0]:
            convs = data[0]['conversations']
            print(f"\n  Conversation fields:")
            for msg in convs:
                print(f"    - Keys: {list(msg.keys())}")
                print(f"      'from' field: {msg.get('from')}")
                print(f"      'role' field: {msg.get('role')}")
else:
    print(f"\n✗ Training file NOT found: {train_path}")

if os.path.exists(val_path):
    with open(val_path, 'r') as f:
        data = json.load(f)
    print(f"\n✓ Validation file exists: {val_path}")
    print(f"  Total samples: {len(data)}")
else:
    print(f"\n✗ Validation file NOT found: {val_path}")

print("\n" + "=" * 60)
