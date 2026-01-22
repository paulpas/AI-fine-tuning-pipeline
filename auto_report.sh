#!/bin/bash
# Auto-reporter for training progress to user

while true; do
    LATEST=$(ls -td finetune/output/python-expert-v5/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        CKPT=$(basename "$LATEST")
        STEP=$(python3 -c "import json; d=json.load(open('$LATEST/trainer_state.json')); print(d.get('global_step', '?'))" 2>/dev/null)
        
        if [ "$STEP" != "?" ] && [ "$STEP" != "1400" ]; then
            # New checkpoint found!
            echo "[$(date '+%H:%M:%S')] 🎯 NEW CHECKPOINT: $CKPT (step $STEP)" >> auto_report.log
        fi
    fi
    sleep 60
done
