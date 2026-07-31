#!/usr/bin/env python3
"""
Download datasets from Hugging Face Hub and other sources.
Supports multiple datasets for DeepSeek fine-tuning.
"""

import os
import sys
import argparse
import yaml
import json
import requests
from pathlib import Path
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


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


def download_from_hf_hub(source, output_dir, cache_dir=None):
    """Download dataset from Hugging Face Hub."""
    try:
        url = source['url']
        filename = os.path.basename(url)
        output_path = os.path.join(output_dir, filename)
        
        # If URL contains huggingface.co, use hf_hub_download
        if 'huggingface.co' in url:
            print(f"  Downloading via Hugging Face Hub...")
            downloaded_path = hf_hub_download(
                repo_id="/".join(url.split("/datasets/")[-1].split("/")[0:2]),
                filename="data/" + filename if "train.json" in url else filename,
                repo_type="dataset",
                cache_dir=cache_dir,
                resume_download=True
            )
            # Copy to output directory
            import shutil
            shutil.copy2(downloaded_path, output_path)
        else:
            # Regular URL download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=f"  {filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        
        print(f"  ✓ Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return None


def download_parquet_dataset(source, output_dir):
    """Download Parquet dataset (OpenCoder format)."""
    try:
        import pandas as pd
        
        url = source['url']
        filename = os.path.basename(url)
        output_path = os.path.join(output_dir, filename)
        
        print(f"  Downloading Parquet file...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"  {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        # Validate Parquet file
        print(f"  Validating Parquet file...")
        df = pd.read_parquet(output_path)
        print(f"  ✓ Retrieved {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        
        return output_path
    except Exception as e:
        print(f"  ✗ Error downloading Parquet dataset: {e}")
        return None


def download_json_dataset(source, output_dir):
    """Download JSON dataset."""
    try:
        import pandas as pd
        
        url = source['url']
        filename = os.path.basename(url)
        output_path = os.path.join(output_dir, filename)
        
        print(f"  Downloading JSON file...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"  {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        # Validate JSON file
        print(f"  Validating JSON file...")
        
        # Try JSON lines format first
        if filename.endswith('.jsonl'):
            df = pd.read_json(output_path, lines=True)
        else:
            # Try regular JSON
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)
        
        print(f"  ✓ Retrieved {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        
        return output_path
    except Exception as e:
        print(f"  ✗ Error downloading JSON dataset: {e}")
        return None


def download_csv_dataset(source, output_dir):
    """Download CSV dataset."""
    try:
        import pandas as pd
        
        url = source['url']
        filename = os.path.basename(url)
        output_path = os.path.join(output_dir, filename)
        
        print(f"  Downloading CSV file...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"  {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        # Validate CSV file
        print(f"  Validating CSV file...")
        df = pd.read_csv(output_path)
        print(f"  ✓ Retrieved {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        
        return output_path
    except Exception as e:
        print(f"  ✗ Error downloading CSV dataset: {e}")
        return None


def verify_dataset_file(filepath, expected_format="json"):
    """Verify downloaded dataset file integrity."""
    try:
        import pandas as pd
        
        print(f"  Verifying: {filepath}")
        
        if expected_format == "parquet":
            df = pd.read_parquet(filepath)
        elif expected_format == "jsonl":
            df = pd.read_json(filepath, lines=True)
        elif expected_format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
        elif expected_format == "csv":
            df = pd.read_csv(filepath)
        else:
            print(f"  ✗ Unknown format: {expected_format}")
            return False
        
        print(f"  ✓ Valid: {len(df)} samples, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download datasets for DeepSeek fine-tuning')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to download (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get dataset configuration
    dataset_config = config.get('dataset', {})
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        io_config = config.get('cloud', {}).get('io')
        output_dir = io_config['dataset_dir']['raw']
    
    print(f"Output directory: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get cache directory
    io_config = config.get('cloud', {}).get('io')
    cache_dir = io_config.get('cache_dir', './cache')
    
    # Determine which datasets to download
    if args.dataset:
        filtered_sources = [s for s in dataset_config.get('sources', []) if s['name'] == args.dataset]
        if not filtered_sources:
            print(f"✗ Dataset '{args.dataset}' not found in configuration")
            return 1
        datasets_to_download = filtered_sources
    else:
        datasets_to_download = dataset_config.get('sources', [])
    
    # Download datasets
    success_count = 0
    for source in datasets_to_download:
        print(f"\n{'=' * 60}")
        print(f"Processing: {source['name']}")
        print(f"Source: {source['url']}")
        print(f"Format: {source['format']}")
        print(f"{'=' * 60}")
        
        dataset_format = source.get('format', 'json')
        
        # Download based on format
        if dataset_format == 'parquet':
            output_path = download_parquet_dataset(source, output_dir)
        elif dataset_format == 'json' or dataset_format == 'jsonl':
            output_path = download_json_dataset(source, output_dir)
        elif dataset_format == 'csv':
            output_path = download_csv_dataset(source, output_dir)
        else:
            print(f"✗ Unknown format: {dataset_format}")
            continue
        
        if output_path:
            # Verify downloaded file
            if verify_dataset_file(output_path, dataset_format):
                success_count += 1
            else:
                print(f"✗ Verification failed for {source['name']}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"Datasets requested: {len(datasets_to_download)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {len(datasets_to_download) - success_count}")
    
    if success_count == len(datasets_to_download):
        print("\n✓ All datasets downloaded successfully!")
        print(f"Files saved to: {output_dir}")
        return 0
    else:
        print(f"\n✗ {len(datasets_to_download) - success_count} download(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())