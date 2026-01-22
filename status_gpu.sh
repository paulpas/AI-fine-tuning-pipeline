#!/bin/bash
# Quick GPU training status checker

echo "=== GPU Training Status ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if training is running
if pgrep -f "axolotl.cli.train" > /dev/null; then
    echo "✓ Training: RUNNING"
    
    # Get training progress
    if [ -f "training_resumed_gpu_live.log" ]; then
        latest=$(tail -1 training_resumed_gpu_live.log 2>/dev/null | grep -oP '\d+%')
        if [ ! -z "$latest" ]; then
            echo "  Progress: $latest (see: tail -f training_resumed_gpu_live.log)"
        fi
    fi
else
    echo "✗ Training: STOPPED"
fi

echo ""
echo "=== GPU Utilization ==="
rocm-smi --csv | tail -4

echo ""
echo "=== Latest Training Steps ==="
tail -15 /tmp/training_resume.log 2>/dev/null | grep -E "it/s|step|loss|Training" | tail -5 || echo "(No recent steps logged)"
