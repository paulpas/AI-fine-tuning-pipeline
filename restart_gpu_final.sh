#!/bin/bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_HOME=/opt/rocm

echo "=========================================="
echo "TRAINING WITH 4 AMD GPUS (ROCm 6.2.4)"
echo "=========================================="
echo "PyTorch: 2.5.1+rocm6.2"
echo "Transformers: 4.40.0 (torch.load compatible)"
echo ""

LOGFILE="training_gpu_$(date +%Y%m%d_%H%M%S).log"
uv run accelerate launch --num_processes=4 -m axolotl.cli.train finetune/axolotl_config_python-expert.yaml --resume_from_checkpoint finetune/output/python-expert-v5/checkpoint-1400/ 2>&1 | tee "$LOGFILE"
