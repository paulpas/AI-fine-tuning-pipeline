#!/bin/bash
# Wrapper script to start auto-recovery training
# Can be run with nohup or in screen/tmux for persistent execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/tmp/training_recovery"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "========================================"
echo "Training Auto-Recovery System"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Script: $SCRIPT_DIR/auto_recover.py"
echo "Logs: $LOG_DIR/recovery_*.log"
echo "========================================"
echo ""

# Check if already running
if pgrep -f "auto_recover.py" > /dev/null; then
    echo "⚠️  Auto-recovery is already running!"
    echo "PID: $(pgrep -f 'auto_recover.py')"
    echo ""
    echo "To stop existing instance:"
    echo "  pkill -f auto_recover.py"
    exit 1
fi

# Check if base config exists
if [ ! -f "$SCRIPT_DIR/axolotl_config_v2_resume.yaml" ]; then
    echo "❌ Config not found: axolotl_config_v2_resume.yaml"
    echo "Please run: ./resume_v2.sh first to generate the config"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$PROJECT_DIR/output/terraform-expert-v2/checkpoint-92" ]; then
    echo "⚠️  Checkpoint not found: output/terraform-expert-v2/checkpoint-92"
    echo "Training will resume from the latest available checkpoint"
fi

echo "Starting auto-recovery training..."
echo "Press Ctrl+C to stop"
echo ""

# Activate venv and run auto-recovery
cd "$PROJECT_DIR"
source .venv/bin/activate

python3 "$SCRIPT_DIR/auto_recover.py"
