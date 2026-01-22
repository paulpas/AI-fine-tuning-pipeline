#!/bin/bash
# Resume terraform-expert-v2 training from checkpoint-92
# Uses conservative settings to avoid GPU hangs

set -e

cd /home/paulpas/git/ideas/llm_training_web_data
source .venv/bin/activate

# ROCm environment for AMD MI50 GPUs
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export GPU_DEVICE_ORDINAL=0,1,2,3

# Reduce memory pressure
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

echo "=============================================="
echo "Resuming terraform-expert-v2 Training"
echo "=============================================="
echo "Config: finetune/axolotl_config_v2_resume.yaml"
echo "Checkpoint: finetune/output/terraform-expert-v2/checkpoint-92"
echo "Changes from original:"
echo "  - sequence_len: 2048 -> 1536"
echo "  - sample_packing: true -> false"
echo "  - micro_batch_size: 2 -> 1"
echo "  - gradient_accumulation_steps: 8 -> 4"
echo "=============================================="

# Resume training from checkpoint
echo ""
echo "Starting training resume..."
accelerate launch -m axolotl.cli.train \
  finetune/axolotl_config_v2_resume.yaml \
  --resume-from-checkpoint finetune/output/terraform-expert-v2/checkpoint-92

echo ""
echo "Training complete!"
echo "Output: finetune/output/terraform-expert-v2/"
