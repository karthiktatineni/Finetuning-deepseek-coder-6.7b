#!/usr/bin/env python3
"""
Download models from Hugging Face Hub.
Supports DeepSeek-Coder and related models.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download, login
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


def download_model(model_name, cache_dir=None, revision="main"):
    """Download model from Hugging Face Hub."""
    print(f"\nDownloading model: {model_name}")
    print(f"This may take a while depending on your internet connection...")
    
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            revision=revision,
            resume_download=True,
            tqdm_class=tqdm
        )
        print(f"✓ Model saved to: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return None


def download_tokenizer(model_name, cache_dir=None, revision="main"):
    """Download tokenizer from Hugging Face Hub."""
    print(f"\nDownloading tokenizer for: {model_name}")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True
        )
        
        # Save tokenizer explicitly
        tokenizer_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        
        print(f"✓ Tokenizer saved to: {tokenizer_path}")
        return tokenizer_path
    except Exception as e:
        print(f"✗ Error downloading tokenizer: {e}")
        return None


def verify_hf_token():
    """Check if Hugging Face token is available."""
    print("\nChecking Hugging Face authentication...")
    
    hf_token = os.environ.get('HF_TOKEN')
    
    if hf_token:
        print(f"✓ HF_TOKEN found (length: {len(hf_token)})")
        try:
            login(token=hf_token)
            print("✓ Logged into Hugging Face Hub")
            return True
        except Exception as e:
            print(f"✗ Error logging in: {e}")
            return False
    else:
        print("⚠ HF_TOKEN not found in environment")
        print("  Some models may require authentication")
        
        # Try downloading anyway
        try:
            login()
            print("✓ Using cached authentication")
            return True
        except:
            print("⚠ Please set HF_TOKEN for full access")
            print("  export HF_TOKEN=your_token_here")
            return True  # Continue anyway


def main():
    parser = argparse.ArgumentParser(description='Download models for DeepSeek fine-tuning')
    parser.add_argument('--config', type=str, default='config/local.yaml',
                        help='Configuration file to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to download (overrides config)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help 'Override cache directory')
    parser.add_argument('--skip-token', action='store_true',
                        help='Skip Hugging Face token verification')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Download Script")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Get cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        # Try to get cache dir from config
        io_config = config.get('local', {}).get('io') or config.get('cloud', {}).get('io')
        cache_dir = io_config.get('cache_dir', './cache')
    
    print(f"\nCache directory: {cache_dir}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Verify HF token if not skipped
    if not args.skip_token:
        if not verify_hf_token():
            print("\n⚠ Continuing without authentication...")
    
    # Determine which models to download
    if args.model:
        models_to_download = [args.model]
    else:
        # Get models from config
        model_config = config.get('model', {})
        base_model = model_config.get('base_model', 'deepseek-ai/deepseek-coder-6.7b-instruct')
        models_to_download = [base_model]
    
    # Download models
    success_count = 0
    for model_name in models_to_download:
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_name}")
        print(f"{'=' * 60}")
        
        # Download model
        model_path = download_model(model_name, cache_dir)
        if model_path:
            success_count += 1
        else:
            print(f"✗ Failed to download {model_name}")
            continue
        
        # Download tokenizer
        tokenizer_path = download_tokenizer(model_name, cache_dir)
        if not tokenizer_path:
            print(f"⚠ Failed to download tokenizer for {model_name}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"Models requested: {len(models_to_download)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {len(models_to_download) - success_count}")
    
    if success_count == len(models_to_download):
        print("\n✓ All models downloaded successfully!")
        return 0
    else:
        print(f"\n✗ {len(models_to_download) - success_count} download(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())