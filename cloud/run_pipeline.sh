#!/usr/bin/env bash
# Complete Staged Pipeline Script for DeepSeek Fine-Tuning up to 500,000 Samples

set -e

echo "================================================================="
echo "  DEEPSEEK 6.7B CONTINUAL STAGED TRAINING PIPELINE (500K TOTAL)  "
echo "================================================================="

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage 2: Train samples 100,000 to 300,000 (Next 200k)
echo ""
echo "[STAGE 2/3] Continual Fine-Tuning on samples 100,000 -> 300,000..."
python training/train_continual.py \
  --config config/cloud_full.yaml \
  --adapter-path outputs/deepskip_tokenization/final_adapter \
  --skip-samples 100000 \
  --num-samples 200000 \
  --batch-size 2 \
  --grad-accum 2 \
  --max-steps 1000 \
  --output-name deepseek_stage2_300k

# Stage 3: Train samples 300,000 to 500,000 (Final 200k -> 500k total)
echo ""
echo "[STAGE 3/3] Continual Fine-Tuning on samples 300,000 -> 500,000..."
python training/train_continual.py \
  --config config/cloud_full.yaml \
  --adapter-path outputs/deepseek_stage2_300k/final_continual_adapter \
  --skip-samples 300000 \
  --num-samples 200000 \
  --batch-size 2 \
  --grad-accum 2 \
  --max-steps 1000 \
  --output-name deepseek_stage3_500k_final

echo ""
echo "================================================================="
echo "  ✓ ALL STAGES COMPLETED SUCCESSFULLY! (500,000 SAMPLES TOTAL)   "
echo "  Final 500k Adapter: outputs/deepseek_stage3_500k_final/final_continual_adapter"
echo "================================================================="
