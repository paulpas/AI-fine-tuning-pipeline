#!/bin/bash
export HIP_VISIBLE_DEVICES=0
export ROCM_HOME=/opt/rocm

LOGFILE="training_single_gpu_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training on GPU 0 (ROCm 6.2.4)"
uv run accelerate launch --num_processes=1 -m axolotl.cli.train finetune/axolotl_config_python-expert.yaml --resume_from_checkpoint finetune/output/python-expert-v5/checkpoint-1400/ 2>&1 | tee "$LOGFILE" &
