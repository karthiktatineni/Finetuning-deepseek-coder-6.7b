# DeepSeek Coder Fine-Tuning Pipeline - Production Grade

A complete, production-ready LLM fine-tuning pipeline for DeepSeek-Coder 6.7B using QLoRA with support for both local and cloud GPU training.

## Features

- **QLoRA + LoRA + 4-bit Quantization**: Memory-efficient fine-tuning
- **Cross-Platform**: Works on RTX 4050 and A100/H100 without code changes
- **Full Dataset Training**: Processes complete datasets, not limited subsets
- **Resume Support**: Automatic checkpoint discovery and resumption
- **Comprehensive Validation**: Pre-training checks for all dependencies
- **Multi-Format Logging**: JSON, CSV, TensorBoard, and WandB
- **Complete Pipeline**: From dataset download to model deployment

## System Requirements

### Local Development (RTX 4050 6GB)
- CUDA 11.8+ or 12.1+
- Python 3.10+
- 16GB+ RAM
- 20GB+ free disk space

### Cloud GPU (A100/H100)
- CUDA 11.8+ or 12.1+
- Python 3.10+
- 120GB+ RAM
- 50GB+ free disk space

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
python scripts/download_models.py --config config/local.yaml
```

### 3. Download Datasets

```bash
python scripts/download_datasets.py --config config/local.yaml
```

### 4. Preprocess Datasets

```bash
# Preprocess individual datasets
python preprocessing/preprocess.py --config config/local.yaml

# Merge all datasets
python preprocessing/merge_datasets.py --config config/local.yaml

# Tokenize merged dataset
python preprocessing/tokenize.py --config config/local.yaml
```

### 5. Train Model

```bash
# Local training
python training/train.py --config config/local.yaml

# Cloud training
python training/train.py --config config/cloud.yaml
```

### 6. Resume Training

```bash
# Auto-resume from latest checkpoint
python training/train.py --config config/local.yaml --resume

# Resume from specific checkpoint
python training/train.py --config config/local.yaml --checkpoint ./checkpoints/checkpoint-5000
```

### 7. Evaluate Model

```bash
python training/evaluate.py --config config/local.yaml --adapter ./adapters/final_adapter
```

### 8. Test Inference

```bash
python inference/chat.py --config config/local.yaml --adapter ./adapters/final_adapter
```

## Configuration System

The pipeline uses a YAML-based configuration system that adapts to different hardware environments:

### Configuration Files

- `config/local.yaml` - Local RTX 4050 configuration
- `config/cloud.yaml` - Cloud A100/H100 configuration
- `config/dataset.yaml` - Dataset sources and processing
- `config/training.yaml` - Training hyperparameters
- `config/model.yaml` - Model and quantization settings

### Switching Environments

```bash
# Local training
python training/train.py --config config/local.yaml

# Cloud training
python training/train.py --config config/cloud.yaml
```

## Project Structure

```
cloud/
├── config/                    # Configuration files
│   ├── local.yaml            # Local GPU settings
│   ├── cloud.yaml            # Cloud GPU settings
│   ├── dataset.yaml          # Dataset configuration
│   ├── training.yaml         # Training parameters
│   └── model.yaml            # Model configuration
│
├── datasets/                  # Data storage
│   ├── raw/                  # Downloaded datasets
│   ├── processed/            # Preprocessed datasets
│   ├── merged/               # Merged dataset
│   └── tokenized/            # Tokenized training data
│
├── scripts/                   # Utility scripts
│   ├── setup.py              # Environment setup
│   ├── download_models.py    # Model downloader
│   ├── download_datasets.py  # Dataset downloader
│   ├── inspect_dataset.py    # Dataset inspector
│   ├── verify_dataset.py     # Dataset verifier
│   └── clean_cache.py        # Cache cleaner
│
├── preprocessing/            # Data preprocessing
│   ├── preprocess.py         # Individual preprocessing
│   ├── merge_datasets.py     # Dataset merging
│   ├── tokenize.py           # Tokenization
│   └── utils.py              # Utility functions
│
├── training/                  # Training scripts
│   ├── train.py              # Main training script
│   ├── resume.py             # Resume training
│   ├── evaluate.py           # Evaluation script
│   ├── export_adapter.py     # Export LoRA adapter
│   ├── merge_lora.py         # Merge LoRA into base model
│   └── trainer_utils.py      # Training utilities
│
├── inference/                # Inference scripts
│   ├── chat.py               # Interactive chat
│   ├── benchmark.py          # Performance benchmarking
│   └── compare_models.py     # Model comparison
│
├── adapters/                 # LoRA adapters output
├── checkpoints/              # Training checkpoints
├── outputs/                  # Model outputs
├── logs/                     # Training logs
├── cache/                    # Hugging Face cache
└── README.md                 # This file
```

## Training Pipeline

### Multi-Epoch Training

The pipeline trains for complete epochs, not fixed steps:

- **Automatic Epoch Completion**: Trains until configured epochs are complete
- **Intelligent Checkpointing**: Saves progress every N steps
- **Auto-Resume**: Continues from interruption
- **Full Dataset**: Uses all processed data, no subsets

### Memory Management

Automatic VRAM adaptation based on hardware:

```python
# Local RTX 4050 (6GB)
- Batch size: 1
- Gradient accumulation: 4
- Sequence length: 1024
- Precision: fp16

# Cloud A100 (40GB)
- Batch size: 4
- Gradient accumulation: 1
- Sequence length: 4096
- Precision: bf16
```

### Resume Training

Automatic resumption with full state restoration:

- Optimizer state
- Scheduler state
- Random number generator state
- Training progress
- Metrics history

## Dataset Pipeline

### Supported Datasets

- OpenCoder Stage 1 & Stage 2
- CodeAlpaca
- ClassEval
- APPS
- CodeSearchNet
- Custom datasets

### Pipeline Stages

1. **Download**: Fetch datasets from Hugging Face
2. **Validate**: Check data integrity and schema
3. **Preprocess**: Normalize to chat format
4. **Merge**: Combine datasets with shuffling
5. **Tokenize**: Pre-tokenize for training efficiency
6. **Split**: Train/validation split

## Model Export

### Export LoRA Adapter

```bash
python training/export_adapter.py --config config/local.yaml --checkpoint ./checkpoints/checkpoint-final
```

### Merge with Base Model

```bash
python training/merge_lora.py --config config/local.yaml --adapter ./adapters/final_adapter --output ./models/merged_model
```

## Monitoring

### Training Metrics

- Loss curves
- Learning rate schedule
- GPU utilization
- Memory usage
- Training speed
- ETA calculation

### Logging Formats

- **TensorBoard**: `tensorboard --logdir logs/`
- **WandB**: Integrated cloud logging
- **JSON**: Detailed metrics history
- **CSV**: Spreadsheet-friendly format

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `per_device_train_batch_size` or decrease `max_seq_length`
2. **Slow Training**: Increase `gradient_accumulation_steps`, decrease `dataloader_num_workers`
3. **Poor Convergence**: Adjust learning rate, warmup steps, or data quality
4. **Resume Issues**: Check log files for error messages, verify checkpoint integrity

### Debug Mode

```bash
# Enable detailed logging
python training/train.py --config config/local.yaml --debug

# Dry run without training
python training/train.py --config config/local.yaml --dry-run
```

## Advanced Usage

### Custom Datasets

Add your dataset to `config/dataset.yaml`:

```yaml
datasets:
  sources:
    - name: "custom_data"
      url: "path/to/your/dataset.json"
      format: "json"
      instruction_field: "prompt"
      response_field: "completion"
```

### Custom Training Parameters

Modify `config/training.yaml` for different training strategies:

```yaml
training:
  num_train_epochs: 5
  learning_rate: 1.0e-4
  warmup_steps: 200
```

### Cloud Deployment

Export and deploy to cloud platforms:

```bash
# Export adapter
python training/export_adapter.py --config config/cloud.yaml

# Upload to cloud storage
aws s3 sync ./adapters s3://my-bucket/adapters/
```

## Requirements

See `requirements.txt` for complete dependency list.

## License

This project follows same license as DeepSeek-Coder model.

## Support

For issues and questions:
- Check logs in `logs/` directory
- Review configuration files
- Validate data integrity
- Monitor GPU utilization