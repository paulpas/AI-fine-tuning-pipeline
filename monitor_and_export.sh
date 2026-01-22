#!/bin/bash
# Monitor training and auto-trigger export when complete

LOG_FILE="training_4gpu_fresh_20260122_0835.log"
CHECKPOINT_DIR="finetune/output/python-expert-v5"
TARGET_STEPS=850

echo "=== TRAINING MONITOR WITH AUTO-EXPORT ==="
echo "Log: $LOG_FILE"
echo "Target: $TARGET_STEPS steps"
echo "Monitoring started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

while true; do
    # Get current step
    LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
    
    # Check for completion
    if echo "$LAST_LINE" | grep -q "Training completed\|Saving final model\|Saving model checkpoint"; then
        echo ""
        echo "✓✓✓ TRAINING COMPLETED ✓✓✓"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        sleep 5  # Wait for final files to be written
        
        echo "Starting EXPORT stage..."
        python -m pipeline.runner --stage export 2>&1 | tee export_$(date '+%Y%m%d_%H%M%S').log
        
        echo ""
        echo "✓ Export completed!"
        echo "Model ready for use with Ollama"
        break
    fi
    
    # Check for GPU hangs
    if tail -3 "$LOG_FILE" 2>/dev/null | grep -q "GPU Hang"; then
        echo "⚠️  GPU HANG DETECTED at $(date '+%H:%M:%S')"
        echo "Last lines:"
        tail -5 "$LOG_FILE"
        break
    fi
    
    # Extract progress
    if echo "$LAST_LINE" | grep -qE "[0-9]+/[0-9]+ \["; then
        STEP_INFO=$(echo "$LAST_LINE" | grep -o "[0-9]\+/$TARGET_STEPS" | head -1)
        if [ ! -z "$STEP_INFO" ]; then
            CURRENT=$(echo "$STEP_INFO" | cut -d/ -f1)
            PERCENT=$((CURRENT * 100 / TARGET_STEPS))
            printf "[%s] Step: %s (%d%%)\n" "$(date '+%H:%M:%S')" "$STEP_INFO" "$PERCENT"
        fi
    fi
    
    sleep 300  # Check every 5 minutes
done
