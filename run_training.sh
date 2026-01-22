#!/bin/bash
export ROCM_HOME=""
cd /home/paulpas/git/ideas/llm_training_web_data
LOGFILE="training_resumed_$(date +%Y%m%d_%H%M%S).log"
echo "===== Training Started =====" > "$LOGFILE"
echo "Time: $(date)" >> "$LOGFILE"
echo "Resuming from checkpoint: finetune/output/python-expert-v5/checkpoint-1400/" >> "$LOGFILE"
echo "Log file: $LOGFILE" >> "$LOGFILE"
echo "" >> "$LOGFILE"
uv run accelerate launch --num_processes=1 -m axolotl.cli.train finetune/axolotl_config_python-expert.yaml --resume_from_checkpoint finetune/output/python-expert-v5/checkpoint-1400/ 2>&1 | tee -a "$LOGFILE"
RET_CODE=$?
echo "" >> "$LOGFILE"
echo "===== Training Finished =====" >> "$LOGFILE"
echo "Exit code: $RET_CODE" >> "$LOGFILE"
echo "Time: $(date)" >> "$LOGFILE"
exit $RET_CODE
