#!/bin/bash
# Train terraform-expert-v2 with improved hyperparameters

set -e

cd /home/paulpas/git/ideas/llm_training_web_data
source .venv/bin/activate

# ROCm environment for AMD MI50 GPUs
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export GPU_DEVICE_ORDINAL=0,1,2,3

echo "=============================================="
echo "Training terraform-expert-v2"
echo "=============================================="
echo "Config: finetune/axolotl_config_v2.yaml"
echo "Dataset: data/training/alpaca_deduped.json (27,873 samples)"
echo "Changes:"
echo "  - Learning rate: 0.0002 -> 0.00005"
echo "  - Epochs: 2 -> 4"
echo "  - LoRA rank: 16 -> 8"
echo "  - Dropout: 0.05 -> 0.1"
echo "  - Weight decay: 0.01 -> 0.05"
echo "  - Max grad norm: 1.0 -> 0.5"
echo "  - Gradient accumulation: 4 -> 8"
echo "=============================================="

# Clear cache
rm -rf finetune/prepared_data_v2

# Start training
accelerate launch -m axolotl.cli.train finetune/axolotl_config_v2.yaml

echo ""
echo "Training complete!"
echo "Output: finetune/output/terraform-expert-v2/"
