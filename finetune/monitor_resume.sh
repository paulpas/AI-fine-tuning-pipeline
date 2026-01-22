#!/bin/bash
# Monitor GPU health during training resume
# Watches for GPU hangs and reports status

INTERVAL=30  # Check every 30 seconds
LOG_FILE="/tmp/resume_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "GPU Health Monitor for Training Resume" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Check interval: ${INTERVAL}s" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo ""

HANG_DETECTED_COUNT=0
LAST_STEP=""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Get GPU status
    GPU_STATUS=$(rocm-smi 2>&1)

    # Check for GPU hangs in recent logs
    HANG_COUNT=$(dmesg 2>/dev/null | grep -c "GPU Hang" || echo "0")

    # Get training process count
    TRAIN_PROCESSES=$(ps aux | grep -E "axolotl|accelerate" | grep -v grep | wc -l)

    # Try to find latest training step from logs
    if [ -f "finetune/output/terraform-expert-v2/training_args.bin" ]; then
        # If we can access training logs, extract recent step
        LAST_STEP=$(tail -n 20 finetune/output/terraform-expert-v2/trainer_state.json 2>/dev/null | grep -o "\"global_step\": [0-9]*" | tail -1 | grep -o "[0-9]*" || echo "?")
    fi

    # Check GPU utilization
    GPU_UTIL=$(rocm-smi 2>&1 | grep -E "^\s*[0-3]" | awk '{print $NF}' | tr '\n' ',' | sed 's/,$//')

    # Extract temperatures
    TEMPS=$(rocm-smi 2>&1 | grep -E "^\s*[0-3]" | awk '{print $(NF-4)}' | tr '\n' ',' | sed 's/,$//')

    # Log status
    {
        echo "[$TIMESTAMP] Status Check"
        echo "  Training processes: $TRAIN_PROCESSES"
        echo "  Current step: $LAST_STEP"
        echo "  GPU utilization: $GPU_UTIL"
        echo "  GPU temperatures: $TEMPS"
        echo "  Hang events detected: $HANG_COUNT"

        # Alert if hangs detected
        if [ "$HANG_COUNT" -gt "$HANG_DETECTED_COUNT" ]; then
            echo "  ⚠️  GPU HANG DETECTED! Count increased from $HANG_DETECTED_COUNT to $HANG_COUNT"
            HANG_DETECTED_COUNT=$HANG_COUNT
        fi

        # Alert if training stopped
        if [ "$TRAIN_PROCESSES" -eq 0 ] && [ "$LAST_STEP" != "?" ]; then
            echo "  ⚠️  TRAINING STOPPED - No training processes running!"
        fi

        # Alert on high temperature
        if echo "$TEMPS" | grep -qE "[7-9][0-9]|[1-9][0-9]{2}"; then
            echo "  ⚠️  HIGH TEMPERATURE WARNING"
        fi

        echo ""
    } | tee -a "$LOG_FILE"

    sleep "$INTERVAL"
done
