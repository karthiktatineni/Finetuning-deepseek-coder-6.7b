#!/usr/bin/env python3
"""
Clean Hugging Face cache and temporary files.
Useful for freeing disk space during development.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def get_hf_cache_size(cache_dir):
    """Calculate total size of Hugging Face cache directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, PermissionError):
                continue
    return total_size


def clean_checkpoint_dir(checkpoint_dir):
    """Clean checkpoints directory."""
    print(f"\nCleaning checkpoints: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"  Checkpoints directory not found")
        return 0
    
    total_size = 0
    cleaned = 0
    
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(item_path)
                          for filename in filenames
                          if os.path.exists(os.path.join(dirpath, filename)))
                
                print(f"  Removing: {item} ({size / 1024 / 1024:.1f} MB)")
                shutil.rmtree(item_path)
                cleaned += 1
                total_size += size
            except Exception as e:
                print(f"  ⚠ Error removing {item}: {e}")
    
    print(f"  ✓ Cleaned {cleaned} checkpoints")
    return total_size


def clean_temp_files(base_dir):
    """Clean temporary files and directories."""
    print(f"\nCleaning temporary files: {base_dir}")
    
    temp_patterns = [
        "*.tmp", "*.temp", "*~", ".DS_Store", "Thumbs.db",
        "__pycache__", "*.pyc", ".pytest_cache"
    ]
    
    total_size = 0
    cleaned = 0
    
    for pattern in temp_patterns:
        for item in Path(base_dir).rglob(pattern):
            try:
                if item.is_file():
                    size = item.stat().st_size
                    item.unlink()
                    cleaned += 1
                    total_size += size
                elif item.is_dir():
                    shutil.rmtree(item)
                    cleaned += 1
            except Exception as e:
                continue
    
    print(f"  ✓ Cleaned {cleaned} temporary items")
    return total_size


def clean_empty_dirs(base_dir):
    """Remove empty directories."""
    print(f"\nCleaning empty directories: {base_dir}")
    
    cleaned = 0
    for item in sorted(Path(base_dir).rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
                cleaned += 1
            except Exception:
                continue
    
    print(f"  ✓ Cleaned {cleaned} empty directories")


def clean_hf_cache(cache_dir, keep_models=None, dry_run=False):
    """Clean Hugging Face cache, optionally keeping specific models."""
    print(f"\nCleaning HF cache: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print(f"  Cache directory not found")
        return 0
    
    if keep_models is None:
        keep_models = []
    
    total_size_before = get_hf_cache_size(cache_dir)
    print(f"  Current cache size: {total_size_before / 1024 / 1024 / 1024:.2f} GB")
    
    total_size_removed = 0
    cleaned = 0
    
    # Scan for model directories
    model_dirs = []
    for item in Path(cache_dir).iterdir():
        if item.is_dir():
            # This might be a model cache directory
            model_dirs.append(item)
    
    print(f"  Found {len(model_dirs)} directories in cache")
    
    for model_dir in model_dirs:
        try:
            # Check if this is a model we want to keep
            model_name = extract_model_name(model_dir)
            
            if model_name and any(keep in model_name.lower() for keep in keep_models):
                print(f"  Keeping: {model_name}")
                continue
            
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(model_dir)
                      for filename in filenames
                      if os.path.exists(os.path.join(dirpath, filename)))
            
            if dry_run:
                print(f"  Would remove: {model_name} ({size / 1024 / 1024:.1f} MB)")
                total_size_removed += size
            else:
                print(f"  Removing: {model_name} ({size / 1024 / 1024:.1f} MB)")
                shutil.rmtree(model_dir)
                cleaned += 1
                total_size_removed += size
                
        except Exception as e:
            print(f"  ⚠ Error processing {model_dir.name}: {e}")
    
    if not dry_run:
        total_size_after = get_hf_cache_size(cache_dir)
        print(f"  ✓ Cleanup complete. New cache size: {total_size_after / 1024 / 1024 / 1024:.2f} GB")
        print(f"  ✓ Saved: {(total_size_before - total_size_after) / 1024 / 1024 / 1024:.2f} GB")
    else:
        print(f"  Dry run: Would save {total_size_removed / 1024 / 1024 / 1024:.2f} GB")
    
    return total_size_removed


def extract_model_name(cache_path):
    """Extract model name from cache path."""
    if 'models--' in cache_path.name:
        return cache_path.name.replace('models--', '').replace('--', '/')
    return None


def clean_dataset_cache(cache_dir):
    """Clean dataset cache."""
    print(f"\nCleaning dataset cache: {cache_dir}")
    
    dataset_cache = Path(cache_dir) / "datasets"
    if not dataset_cache.exists():
        print(f"  Dataset cache not found")
        return 0
    
    total_size = 0
    cleaned = 0
    
    try:
        size = sum(os.path.getsize(os.path.join(dirpath, filename))
                  for dirpath, dirnames, filenames in os.walk(dataset_cache)
                  for filename in filenames
                  if os.path.exists(os.path.join(dirpath, filename)))
        
        print(f"  Removing dataset cache ({size / 1024 / 1024:.1f} MB)")
        
        if not args.dry_run:
            shutil.rmtree(dataset_cache)
            cleaned += 1
            total_size = size
            print(f"  ✓ Cleaned dataset cache")
        
    except Exception as e:
        print(f"  ⚠ Error cleaning dataset cache: {e}")
    
    return total_size


def main():
    parser = argparse.ArgumentParser(description='Clean cache directories and temporary files')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Hugging Face cache directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoints directory to clean')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory for temp file cleanup')
    parser.add_argument('--keep-models', type=str, nargs='*', default=[],
                        help='Models to keep in cache (partial matches)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be cleaned without actually cleaning')
    parser.add_argument('--datasets-only', action='store_true',
                        help='Only clean dataset cache')
    parser.add_argument('--models-only', action='store_true',
                        help='Only clean model cache')
    
    global args
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cache Cleanup Tool")
    print("=" * 60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted")
    
    total_size_saved = 0
    
    # Clean Hugging Face cache
    if not args.datasets_only:
        total_size_saved += clean_hf_cache(args.cache_dir, args.keep_models, args.dry_run)
    
    # Clean dataset cache
    if not args.models_only:
        total_size_saved += clean_dataset_cache(args.cache_dir)
    
    # Clean checkpoints
    if not args.dry_run:
        total_size_saved += clean_checkpoint_dir(args.checkpoint_dir)
    else:
        print("\nSkipping checkpoint dry run (use without --dry-run to see checkpoint sizes)")
    
    # Clean temporary files
    if not args.dry_run and not (args.datasets_only or args.models_only):
        total_size_saved += clean_temp_files(args.base_dir)
    
    # Clean empty directories
    if not args.dry_run and not (args.datasets_only or args.models_only):
        clean_empty_dirs(args.base_dir)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Cleanup Summary")
    print(f"{'=' * 60}")
    
    if args.dry_run:
        print(f"Would save: {total_size_saved / 1024 / 1024 / 1024:.2f} GB")
    else:
        print(f"Space saved: {total_size_saved / 1024 / 1024 / 1024:.2f} GB")
        print("✓ Cache cleanup complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())