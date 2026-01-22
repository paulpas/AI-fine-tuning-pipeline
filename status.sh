#!/bin/bash
# Quick training status checker

OUTPUT_DIR="finetune/output/python-expert-v5"
LOGFILE=$(ls -t training_resumed_*.log 2>/dev/null | head -1)

# Status indicator
if ps aux | grep -E "axolotl|accelerate" | grep -v grep > /dev/null 2>&1; then
    STATUS="🟢 RUNNING"
else
    STATUS="🔴 STOPPED"
fi

LATEST=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    CKPT=$(basename "$LATEST")
    STEP=$(python3 -c "import json; d=json.load(open('$LATEST/trainer_state.json')); print(d.get('global_step', '?'))" 2>/dev/null)
    LOSS=$(python3 -c "import json; d=json.load(open('$LATEST/trainer_state.json')); logs=[x for x in d.get('log_history',[]) if 'loss' in x]; print(f\"{logs[-1]['loss']:.4f}\" if logs else '?')" 2>/dev/null)
    PERCENT=$((STEP * 100 / 1758))
    
    echo "📊 Training Status: $STATUS"
    echo "Checkpoint: $CKPT"
    echo "Progress: [$((PERCENT / 5))============================]"
    echo "          $STEP/1758 steps ($PERCENT%)"
    echo "Loss: $LOSS"
    echo ""
    echo "View logs: tail -f $LOGFILE"
else
    echo "Status: $STATUS (no checkpoints yet)"
fi
