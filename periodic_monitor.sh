#!/bin/bash
# Periodic Training Monitor - Reports every 30 minutes

LOGDIR="/home/paulpas/git/ideas/llm_training_web_data"
OUTPUT_DIR="$LOGDIR/finetune/output/python-expert-v5"
MONITOR_LOG="$LOGDIR/monitoring.log"
INTERVAL=1800  # 30 minutes in seconds

# Initialize monitoring log
echo "=== Training Monitor Started ===" >> "$MONITOR_LOG"
echo "Time: $(date)" >> "$MONITOR_LOG"
echo "" >> "$MONITOR_LOG"

while true; do
    {
        echo "=========================================="
        echo "Status Update: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        
        # Check if training is running
        if ps aux | grep -E "axolotl|accelerate" | grep -v grep > /dev/null 2>&1; then
            echo "✅ Training: ACTIVE"
            
            # Get latest checkpoint
            LATEST=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
            if [ -n "$LATEST" ]; then
                CKPT=$(basename "$LATEST")
                CKPT_TIME=$(stat -c "%y" "$LATEST" 2>/dev/null | cut -d' ' -f1-2)
                echo "Latest: $CKPT (updated: $CKPT_TIME)"
                
                # Extract training metrics
                if [ -f "$LATEST/trainer_state.json" ]; then
                    STEP=$(python3 -c "import json; d=json.load(open('$LATEST/trainer_state.json')); print(d.get('global_step', '?'))" 2>/dev/null)
                    LOSS=$(python3 -c "import json; d=json.load(open('$LATEST/trainer_state.json')); logs=[x for x in d.get('log_history',[]) if 'loss' in x]; print(f\"{logs[-1]['loss']:.4f}\" if logs else '?')" 2>/dev/null)
                    PERCENT=$((STEP * 100 / 1758))
                    REMAINING=$((1758 - STEP))
                    
                    echo "Progress: Step $STEP/1758 ($PERCENT%)"
                    echo "Loss: $LOSS"
                    echo "Remaining: $REMAINING steps (~$((REMAINING / 100)) checkpoints)"
                fi
            fi
            
            # Memory status
            MEM=$(ps aux | grep -E "python.*axolotl" | grep -v grep | awk '{sum+=$6} END {print sum " MB"}')
            echo "Memory: $MEM"
            
        else
            echo "❌ Training: STOPPED"
            
            # Check if it finished or crashed
            FINAL=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
            if [ -n "$FINAL" ]; then
                FINAL_STEP=$(python3 -c "import json; d=json.load(open('$FINAL/trainer_state.json')); print(d.get('global_step', '?'))" 2>/dev/null)
                echo "Final checkpoint: $(basename $FINAL) at step $FINAL_STEP"
            fi
        fi
        
        echo "=========================================="
        echo ""
    } | tee -a "$MONITOR_LOG"
    
    sleep $INTERVAL
done
