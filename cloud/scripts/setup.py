#!/usr/bin/env python3
"""
Setup script for DeepSeek Fine-tuning Pipeline.
Verifies system requirements and creates necessary directories.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("  Required: Python 3.10+")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda} is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA is not available")
            return False
    except ImportError:
        print("✗ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking required packages...")
    
    required_packages = [
        'torch', 'transformers', 'peft', 'bitsandbytes', 
        'accelerate', 'datasets', 'trl', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def create_directories(config_file):
    """Create necessary directories based on configuration."""
    print(f"\nCreating directories from {config_file}...")
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Determine which config file is being used
        if 'local' in config:
            io_config = config['local']['io']
        elif 'cloud' in config:
            io_config = config['cloud']['io']
        else:
            print(f"✗ Unknown configuration format in {config_file}")
            return False
        
        # Create directories
        directories = [
            io_config['cache_dir'],
            io_config['output_dir'],
            io_config['adapter_dir'],
            io_config['checkpoint_dir'],
            io_config['log_dir'],
        ]
        
        # Add dataset directories
        for dir_type, dir_path in io_config['dataset_dir'].items():
            directories.append(dir_path)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating directories: {e}")
        return False


def check_disk_space(config_file):
    """Check if there's enough disk space."""
    print("\nChecking disk space...")
    
    try:
        import shutil
        import yaml
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get minimum disk space requirement
        if 'local' in config:
            min_space = config['local']['validation']['min_disk_space']
        elif 'cloud' in config:
            min_space = config['cloud']['validation']['min_disk_space']
        else:
            min_space = 50  # Default 50GB
        
        # Get current directory disk space
        disk_usage = shutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024 ** 3)
        
        if free_space_gb >= min_space:
            print(f"✓ {free_space_gb:.1f} GB free (required: {min_space} GB)")
            return True
        else:
            print(f"✗ {free_space_gb:.1f} GB free (required: {min_space} GB)")
            return False
            
    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return False


def install_dependencies():
    """Install missing dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Setup DeepSeek Fine-tuning Pipeline')
    parser.add_argument('--config', type=str, default='config/local.yaml',
                        help='Configuration file to use')
    parser.add_argument('--install', action='store_true',
                        help='Install missing dependencies')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepSeek Fine-tuning Pipeline Setup")
    print("=" * 60)
    
    results = []
    
    # Run checks unless skipped
    if not args.skip_checks:
        results.append(("Python Version", check_python_version()))
        results.append(("CUDA", check_cuda()))
        results.append(("Dependencies", check_dependencies()))
        
        # Only check directory creation if dependencies are available
        if any(r[1] for r in results):
            results.append(("Disk Space", check_disk_space(args.config)))
            
            # Only create directories if YAML is available
            if check_python_version():
                results.append(("Directories", create_directories(args.config)))
    
    # Install dependencies if requested
    if args.install:
        results.append(("Install Dependencies", install_dependencies()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    for check_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{check_name:20} {status}")
    
    # Overall result
    if all(r[1] for r in results):
        print("\n✓ Setup completed successfully!")
        return 0
    else:
        print("\n✗ Setup failed. Please fix the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())