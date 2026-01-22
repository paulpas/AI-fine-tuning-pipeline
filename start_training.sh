#!/bin/bash
cd /home/paulpas/git/ideas/llm_training_web_data
export ROCM_HOME=""
LOGFILE="training_resumed_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training, logging to $LOGFILE"
uv run accelerate launch --num_processes=1 -m axolotl.cli.train finetune/axolotl_config_python-expert.yaml --resume_from_checkpoint finetune/output/python-expert-v5/checkpoint-1400/ 2>&1 | tee "$LOGFILE"
