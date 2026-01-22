#!/bin/bash
# Training Monitor - Reports progress periodically

LOGDIR="/home/paulpas/git/ideas/llm_training_web_data"
LOGFILE=$(ls -t "$LOGDIR"/training_resumed_*.log 2>/dev/null | head -1)
OUTPUT_DIR="$LOGDIR/finetune/output/python-expert-v5"
CHECKPOINT_DIR="$OUTPUT_DIR"

echo "=========================================="
echo "TRAINING MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Check if process is running
if ps aux | grep -E "axolotl|accelerate" | grep -v grep > /dev/null; then
    echo "✅ Training process: RUNNING"
else
    echo "❌ Training process: NOT RUNNING"
fi

echo ""
echo "--- Checkpoint Status ---"
LATEST=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    CKPT_NAME=$(basename "$LATEST")
    CKPT_TIME=$(stat -c "%y" "$LATEST" | cut -d' ' -f1-2)
    echo "Latest checkpoint: $CKPT_NAME"
    echo "Created: $CKPT_TIME"
    
    # Try to extract step from trainer_state.json
    if [ -f "$LATEST/trainer_state.json" ]; then
        STEP=$(python3 -c "import json; data=json.load(open('$LATEST/trainer_state.json')); print(data.get('global_step', 'unknown'))" 2>/dev/null)
        LOSS=$(python3 -c "import json; data=json.load(open('$LATEST/trainer_state.json')); logs=[x for x in data.get('log_history',[]) if 'loss' in x]; print(f\"{logs[-1]['loss']:.4f}\" if logs else 'N/A')" 2>/dev/null)
        echo "Step: $STEP / 1758 ($(( STEP * 100 / 1758 ))%)"
        echo "Loss: $LOSS"
    fi
else
    echo "No checkpoints found yet (still training first 100 steps)"
fi

echo ""
echo "--- Log Activity (Last 10 Lines) ---"
if [ -f "$LOGFILE" ]; then
    tail -10 "$LOGFILE" | grep -v "^\s*$"
else
    echo "No log file found"
fi

echo ""
echo "--- Memory Usage ---"
ps aux | grep -E "axolotl|accelerate" | grep -v grep | awk '{printf "PID %s: %.1f%% CPU, %dM RAM\n", $2, $3, $6}'

echo ""
echo "--- Estimated Time Remaining ---"
if [ -n "$STEP" ] && [ "$STEP" != "unknown" ]; then
    REMAINING=$((1758 - STEP))
    # Rough estimate: ~15 seconds per step on CPU
    HOURS=$((REMAINING * 15 / 3600))
    MINS=$(((REMAINING * 15 % 3600) / 60))
    echo "Steps remaining: $REMAINING"
    echo "ETA: ~${HOURS}h ${MINS}m"
else
    echo "ETA: ~5-7 hours (from checkpoint-1400)"
fi

echo "=========================================="
