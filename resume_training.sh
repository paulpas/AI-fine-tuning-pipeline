#!/bin/bash
# Resume Training - GUARANTEED Resume from Latest Checkpoint
# This script ensures training resumes from the latest checkpoint or fails clearly

set -e

MODEL_NAME="${1:-python-expert-v5}"
CONFIG="${2:-finetune/axolotl_config_python-expert.yaml}"
OUTPUT_DIR="finetune/output/${MODEL_NAME}"

echo "=========================================="
echo "Resume Training - ${MODEL_NAME}"
echo "=========================================="
echo ""

# Verify output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: Output directory not found: $OUTPUT_DIR"
    echo "Available models:"
    ls -d finetune/output/*/
    exit 1
fi

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ ERROR: No checkpoints found in $OUTPUT_DIR"
    echo "Cannot resume - no checkpoint to resume from"
    exit 1
fi

# Extract checkpoint number
CHECKPOINT_DIR=$(basename "$LATEST_CHECKPOINT")
CHECKPOINT_NUM=$(echo "$CHECKPOINT_DIR" | sed 's/checkpoint-//')

# Get training state to show progress
TRAINER_STATE="$LATEST_CHECKPOINT/trainer_state.json"
if [ ! -f "$TRAINER_STATE" ]; then
    echo "❌ ERROR: trainer_state.json not found in checkpoint"
    exit 1
fi

# Extract current step and total steps
CURRENT_STEP=$(python3 -c "import json; d=json.load(open('$TRAINER_STATE')); print(d.get('global_step', 0))")
TOTAL_STEPS=$(python3 -c "import json; d=json.load(open('$TRAINER_STATE')); print(d.get('max_steps', 'unknown'))")
CURRENT_LOSS=$(python3 -c "import json; d=json.load(open('$TRAINER_STATE')); logs=[x for x in d.get('log_history',[]) if 'loss' in x]; print(logs[-1]['loss'] if logs else 'N/A')" 2>/dev/null || echo "N/A")

# Calculate progress
if [ "$TOTAL_STEPS" != "unknown" ]; then
    PROGRESS=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    REMAINING=$((TOTAL_STEPS - CURRENT_STEP))
else
    PROGRESS="unknown"
    REMAINING="unknown"
fi

echo "✓ Found checkpoint: $CHECKPOINT_DIR"
echo "  Current step: $CURRENT_STEP / $TOTAL_STEPS"
echo "  Progress: $PROGRESS%"
echo "  Remaining: $REMAINING steps"
echo "  Current loss: $CURRENT_LOSS"
echo ""

# Verify config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Starting training with EXPLICIT RESUME..."
echo "  Resume from: $LATEST_CHECKPOINT"
echo "  Config: $CONFIG"
echo ""

# Run training with explicit resume flag
# Both --resume_from_checkpoint and allowing Axolotl to auto-detect
uv run accelerate launch \
    --num_processes=4 \
    -m axolotl.cli.train \
    "$CONFIG" \
    --resume_from_checkpoint "$LATEST_CHECKPOINT" \
    2>&1 | tee -a "training_resume_$(date +%Y%m%d_%H%M%S).log"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully"
else
    echo ""
    echo "❌ Training failed with exit code: $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi
