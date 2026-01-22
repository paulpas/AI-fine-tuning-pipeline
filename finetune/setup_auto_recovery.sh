#!/bin/bash
# Setup script for auto-recovery systemd service
# Run this with: sudo ./setup_auto_recovery.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/training-recovery.service"
SERVICE_NAME="training-recovery"
SYSTEM_SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME.service"

echo "========================================"
echo "Training Auto-Recovery Service Setup"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

# Verify service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Service file not found: $SERVICE_FILE"
    exit 1
fi

echo "✓ Installing systemd service..."
cp "$SERVICE_FILE" "$SYSTEM_SERVICE_PATH"
chmod 644 "$SYSTEM_SERVICE_PATH"

echo "✓ Reloading systemd configuration..."
systemctl daemon-reload

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To start the training service:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable $SERVICE_NAME"
echo ""
echo "To check service status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "To stop the service:"
echo "  sudo systemctl stop $SERVICE_NAME"
echo ""
echo "To restart the service:"
echo "  sudo systemctl restart $SERVICE_NAME"
echo ""
