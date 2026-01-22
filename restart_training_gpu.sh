#!/bin/bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_HOME=/opt/rocm
cd /home/paulpas/git/ideas/llm_training_web_data

echo "=========================================="
echo "RESTARTING TRAINING WITH GPU SUPPORT"
echo "=========================================="
echo "GPUs: 4x AMD Radeon Graphics"
echo "ROCm: 6.2"
echo "PyTorch: 2.5.1+rocm6.2"
echo ""

LOGFILE="training_resumed_gpu_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOGFILE"
echo "Time: $(date)"
echo ""

uv run accelerate launch --num_processes=4 -m axolotl.cli.train finetune/axolotl_config_python-expert.yaml --resume_from_checkpoint finetune/output/python-expert-v5/checkpoint-1400/ 2>&1 | tee "$LOGFILE"

RET=$?
echo ""
echo "Training completed with exit code: $RET"
exit $RET
