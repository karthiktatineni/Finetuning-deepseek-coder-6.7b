# DeepSeek Fine-Tuning Pipeline - Complete Execution Guide

This is the complete sequence of Python files to execute for production-grade LLM fine-tuning.

## Pipeline Execution Sequence

### 1. Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script to verify environment
python scripts/setup.py --config config/cloud.yaml
```

### 2. Model Download
```bash
# Download DeepSeek-Coder model
python scripts/download_models.py --config config/cloud.yaml
```

### 3. Dataset Preparation
```bash
# Download all datasets
python scripts/download_datasets.py --config config/cloud.yaml

# Inspect dataset structure (optional but recommended)
python scripts/inspect_dataset.py --dataset datasets/raw/[your_dataset_file]

# Verify dataset quality (recommended before preprocessing)
python scripts/verify_dataset.py --dataset datasets/raw/[your_dataset_file] --instruction instruction --response response
```

### 4. Data Preprocessing
```bash
# Preprocess each downloaded dataset individually
python preprocessing/preprocess_simple.py --config config/cloud.yaml --input datasets/raw/opencoder_stage1
python preprocessing/preprocess_simple.py --config config/cloud.yaml --input datasets/raw/opencoder_stage2
python preprocessing/preprocess_simple.py --config config/cloud.yaml --input datasets/raw/codealpaca
python preprocessing/preprocess_simple.py --config config/cloud.yaml --input datasets/raw/apps
python preprocessing/preprocess_simple.py --config config/cloud.yaml --input datasets/raw/codesearchnet
# Note: classeval is eval-only and should not be preprocessed for training
```

### 5. Dataset Merging
```bash
# Merge all preprocessed datasets into single training set
python preprocessing/merge_datasets.py --config config/cloud.yaml

# This creates:
# - datasets/merged/train.json (95% of data)
# - datasets/merged/validation.json (5% of data)
# - datasets/merged/merged.json (all data)
```

### 6. Tokenization
```bash
# Tokenize the merged datasets for efficient training
python preprocessing/tokenize.py --config config/cloud.yaml

# This creates:
# - datasets/tokenized/train/ (HuggingFace Dataset format)
# - datasets/tokenized/validation/ (HuggingFace Dataset format)
# - datasets/tokenized/metadata.json (tokenization metadata)
```

### 7. Training
```bash
# Start full training with all datasets
python training/train.py --config config/cloud.yaml --output-name deepseek_full_training

# Resume training from latest checkpoint if interrupted
python training/train.py --config config/cloud.yaml --resume

# Resume from specific checkpoint
python training/train.py --config config/cloud.yaml --checkpoint checkpoints/checkpoint-5000
```

### 8. Evaluation
```bash
# List available benchmarks
python evaluation/evaluate.py --benchmark-list

# Run comprehensive evaluation (30+ benchmarks)
python evaluation/evaluate.py --config config/cloud.yaml --adapter adapters/final_adapter

# Run specific benchmarks
python evaluation/evaluate.py --config config/cloud.yaml --benchmarks python_basics javascript_basics algorithms_sorting

# Run with limited prompts per benchmark for quick testing
python evaluation/evaluate.py --config config/cloud.yaml --max-prompts 5

# Run and save results to custom location
python evaluation/evaluate.py --config config/cloud.yaml --output results/custom_results.json
```

### 9. Inference (Optional)
```bash
# Interactive testing with trained model
python inference/chat.py --config cloud.yaml --adapter adapters/final_adapter

# Benchmark performance comparison
python inference/benchmark.py --config cloud.yaml

# Compare models
python inference/compare_models.py --model1 adapters/final_adapter --model2 adapters/another_adapter
```

### 10. Cleanup (Optional)
```bash
# Clean cache and temporary files
python scripts/clean_cache.py --cache-dir ./cache

# Clean only dataset cache
python scripts/clean_cache.py --datasets-only

# Dry run to see what would be cleaned
python scripts/clean_cache.py --dry-run

# Keep specific models while cleaning others
python scripts/clean_cache.py --keep-models deepseek-coder
```

## File Dependencies

### Pre-requisites
- `requirements.txt` - DEPENDENCY INSTALLATION
- `config/cloud.yaml` - MAIN CONFIGURATION FILE

### Stage 1: Setup
- `scripts/setup.py` → Environment validation

### Stage 2: Model & Data  
- `scripts/download_models.py` → Model downloading
- `scripts/download_datasets.py` → Dataset downloading

### Stage 3: Data Processing
- `scripts/inspect_dataset.py` → Dataset inspection (optional)
- `scripts/verify_dataset.py` → Dataset validation (recommended)
- `preprocessing/preprocess.py` → Individual dataset preprocessing
- `preprocessing/merge_datasets.py` → Dataset merging
- `preprocessing/tokenize.py` → Tokenization
- `preprocessing/utils.py` → Utility functions (imported)

### Stage 4: Training
- `training/train.py` → Main training script

### Stage 5: Evaluation
- `evaluation/evaluate.py` → Comprehensive benchmarking

### Stage 6: Inference (Optional)
- `inference/chat.py` → Interactive testing

## Configuration Strategy

The entire pipeline uses a single configuration system:

```bash
# Use configuration file for all operations
python [script] --config config/cloud.yaml
```

All paths, parameters, and settings are loaded from the config file - no manual path editing needed.

## Output Locations

### Model Files
- Base Model: `cache/models--deepseek-ai--deepseek-coder-6.7b-instruct/`
- Adapters: `adapters/`
- Checkpoints: `checkpoints/`

### Data Files
- Raw Data: `datasets/raw/`
- Processed Data: `datasets/processed/`
- Merged Data: `datasets/merged/`
- Tokenized Data: `datasets/tokenized/`

### Results & Logs
- Training Logs: `logs/deepseek_full_training_[timestamp].log`
- Results: `results/evaluation_results.json`
- Outputs: `outputs/`

## Quick Start Summary

### Minimal Test Run (Single Dataset)
```bash
python scripts/download_models.py --config config/cloud.yaml
python preprocessing/preprocess.py --config config/cloud.yaml --input datasets/raw/codealpaca.json
python preprocessing/tokenize.py --config cloud.yaml --train-file datasets/processed/codealpaca_processed.json
python training/train.py --config cloud.yaml --output-name test_run
python evaluation/evaluate.py --config cloud.yaml --benchmarks python_basics --max-prompts 3
```

### Full Production Pipeline
```bash
python scripts/setup.py --config config/cloud.yaml
python scripts/download_models.py --config config/cloud.yaml
python scripts/download_datasets.py --config config/cloud.yaml

# Preprocess all datasets
for file in datasets/raw/*.json datasets/raw/*.parquet; do
    python preprocessing/preprocess.py --config config/cloud.yaml --input "$file"
done

python preprocessing/merge_datasets.py --config config/cloud.yaml
python preprocessing/tokenize.py --config config/cloud.yaml
python training/train.py --config config/cloud.yaml --output-name production_run
python evaluation/evaluate.py --config config/cloud.yaml
```

## Error Recovery

### Resume Interrupted Training
```bash
# Automatically resume from latest checkpoint
python training/train.py --config cloud.yaml --resume
```

### Retry Failed Pipeline Stage
```bash
# Each stage can be independently rerun
python preprocessing/merge_datasets.py --config cloud.yaml
```

### Validation Issues
```bash
# Inspect problematic dataset
python scripts/inspect_dataset.py --dataset datasets/raw/problematic_data.json

# Reprocess specific dataset
python preprocessing/preprocess.py --config cloud.yaml --input datasets/raw/problematic_data.json
```

## Monitoring Progress

### Check Training Progress
```bash
# View training logs
tail -f logs/deepseek_full_training_[timestamp].log

# Check checkpoints
ls -la checkpoints/

# Monitor GPU usage
nvidia-smi
```

### Check Evaluation Results
```bash
# View summary
python -c "import json; print(json.load(open('results/evaluation_results.json'))['summary'])"
```

This execution guide ensures production-grade, reproducible fine-tuning from start to finish.





next stages



Here are the exact **Bash commands** for your SSH Linux cloud environment:

---

### ⚡ Option 1: Run Stage 2 & Stage 3 Automatically (Bash Script)

Run this single command in your Linux SSH terminal:

```bash
chmod +x run_pipeline.sh && ./run_pipeline.sh
```

---

### 📌 Option 2: Run Each Stage Manually in Bash

#### 🔹 STAGE 2 (Samples 100,000 $\rightarrow$ 300,000):
```bash
WANDB_MODE=disabled python training/train_continual.py \
  --config config/cloud_full.yaml \
  --adapter-path outputs/deepskip_tokenization/final_adapter \
  --skip-samples 100000 \
  --num-samples 200000 \
  --batch-size 2 \
  --grad-accum 2 \
  --max-steps 1000 \
  --output-name deepseek_stage2_300k
```

---

#### 🔹 STAGE 3 (Samples 300,000 $\rightarrow$ 500,000):
```bash
WANDB_MODE=disabled python training/train_continual.py \
  --config config/cloud_full.yaml \
  --adapter-path outputs/deepseek_stage2_300k/final_continual_adapter \
  --skip-samples 300000 \
  --num-samples 200000 \
  --batch-size 2 \
  --grad-accum 2 \
  --max-steps 1000 \
  --output-name deepseek_stage3_500k_final
```

---

### 📥 Download Final 500k Adapter to your local PC:
After Stage 3 finishes, run this command from your **local PC terminal**:

```bash
scp -r user@<CLOUD_IP>:/path/to/Finetuning-deepseek-coder-6.7b/cloud/outputs/deepseek_stage3_500k_final/final_continual_adapter ./adapters/
```