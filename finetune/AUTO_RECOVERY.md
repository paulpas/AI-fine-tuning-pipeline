# Auto-Recovery Training System

Automatic detection and recovery from GPU hangs during LLM fine-tuning.

## Overview

The auto-recovery system monitors training in real-time and automatically:
- Detects GPU hangs via kernel logs
- Detects training process crashes
- Detects progress stalls (training not advancing steps)
- Kills zombie processes
- Resets GPU state
- Resumes from last checkpoint with progressively more conservative settings
- Logs all events for debugging

## Recovery Levels

If training fails, recovery attempts increasingly conservative configurations:

| Level | seq_len | batch | grad_acc | learning_rate |
|-------|---------|-------|----------|---------------|
| 0 | 1536 | 1 | 4 | 0.00005 |
| 1 | 1024 | 1 | 2 | 0.00003 |
| 2 | 768 | 1 | 1 | 0.00002 |
| 3 | 512 | 1 | 1 | 0.00001 |

Each level is more conservative to avoid future GPU hangs.

## Quick Start

### Option 1: Standalone (No systemd)

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Make scripts executable
chmod +x finetune/start_auto_recovery.sh
chmod +x finetune/auto_recover.py

# Start training with auto-recovery
finetune/start_auto_recovery.sh
```

This runs in foreground. Use `nohup` or `screen`/`tmux` for background:

```bash
# With nohup
nohup finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &

# With screen
screen -S training -d -m finetune/start_auto_recovery.sh
screen -r training    # Reconnect
```

### Option 2: Systemd Service (Recommended)

Runs as a system service that persists across reboots and can be monitored.

**Installation:**
```bash
cd /home/paulpas/git/ideas/llm_training_web_data/finetune

# Install the service
sudo ./setup_auto_recovery.sh

# Start the service
sudo systemctl start training-recovery

# Enable auto-start on boot
sudo systemctl enable training-recovery
```

**Management:**
```bash
# Check status
sudo systemctl status training-recovery

# View logs (real-time)
sudo journalctl -u training-recovery -f

# View last 100 lines
sudo journalctl -u training-recovery -n 100

# Restart service
sudo systemctl restart training-recovery

# Stop service
sudo systemctl stop training-recovery
```

## How It Works

### 1. Startup
```
1. Activate venv
2. Set up ROCm environment
3. Clean up any zombie processes
4. Start initial training
```

### 2. Monitoring Loop (every 10 seconds)
```
1. Check for new GPU hangs in kernel logs
2. Check if training process is still running
3. Check if training is making progress (steps advancing)
4. If healthy: sleep and repeat
5. If failure detected: initiate recovery
```

### 3. Recovery Procedure
```
1. Log failure reason
2. Kill zombie processes
3. Reset GPU state with rocm-smi
4. Create recovery config with more conservative settings
5. Resume from checkpoint-92 with new config
6. Reset monitoring state
7. Resume monitoring loop
```

### 4. Maximum Attempts
Allows up to 5 recovery attempts before giving up. After 5 failures:
- Stops training
- Logs final error
- Exits with error code

## Logs

Logs are stored in `/tmp/training_recovery/` with timestamps:

```
/tmp/training_recovery/
├── recovery_20260122_170000.log   # Recovery attempt 1
├── recovery_20260122_171000.log   # Recovery attempt 2
└── recovery_20260122_172000.log   # Recovery attempt 3
```

Each log file contains:
- Timestamp of each check
- GPU hang detections
- Process status
- Training step progress
- Recovery actions taken

**View recovery logs:**
```bash
# Latest log
tail -f /tmp/training_recovery/recovery_*.log | sort -u

# All logs
cat /tmp/training_recovery/*.log

# Search for failures
grep -i "failure\|hang\|error" /tmp/training_recovery/*.log
```

## Monitoring During Training

While training is running, check status:

```bash
# Overall status
ps aux | grep auto_recover

# GPU usage
rocm-smi

# Temperature
rocm-smi | grep -E "C|GPU"

# Current training step
tail -f /tmp/training_recovery/recovery_*.log
```

For systemd service:
```bash
# Real-time logs
sudo journalctl -u training-recovery -f

# Search logs
sudo journalctl -u training-recovery | grep "hang\|failure\|recovery"
```

## Troubleshooting

### Service won't start

Check the service file path:
```bash
ls -la /etc/systemd/system/training-recovery.service

# If missing, reinstall
cd /home/paulpas/git/ideas/llm_training_web_data/finetune
sudo ./setup_auto_recovery.sh
```

### Check logs for errors
```bash
# Systemd logs
sudo journalctl -u training-recovery -n 50

# Recovery logs
tail -20 /tmp/training_recovery/recovery_*.log
```

### Still getting GPU hangs?

The recovery system should handle them automatically, but if hangs persist:

1. **Manual GPU reset:**
   ```bash
   sudo rocm-smi --gpureset
   sleep 5
   ```

2. **More conservative settings:** Manually edit a recovery level in `auto_recover.py`

3. **System reboot:** (last resort)
   ```bash
   sudo reboot
   ```

### Service not auto-starting on boot

```bash
# Enable service
sudo systemctl enable training-recovery

# Verify it's enabled
sudo systemctl is-enabled training-recovery
# Should output: enabled
```

## Configuration

To modify recovery behavior, edit `/finetune/auto_recover.py`:

**Change max recovery attempts:**
```python
CONFIG = {
    'max_recovery_attempts': 5,  # Change this
    ...
}
```

**Modify recovery levels:**
```python
RECOVERY_LEVELS = [
    {
        'name': 'Level 0',
        'sequence_len': 1536,      # Adjust these
        'micro_batch_size': 1,
        ...
    },
    ...
]
```

**Change monitoring intervals:**
```python
CONFIG = {
    'hang_check_interval': 30,      # seconds
    'process_check_interval': 10,   # seconds
    ...
}
```

## Performance Impact

Recovery levels trade training speed for stability:

- **Level 0**: ~90% of original speed (recommended for most)
- **Level 1**: ~70% of original speed (conservative)
- **Level 2**: ~50% of original speed (very conservative)
- **Level 3**: ~30% of original speed (ultra conservative)

The system automatically escalates through levels on repeated failures.

## Example Output

```
[2026-01-22 17:00:00] Training Auto-Recovery System Started
[2026-01-22 17:00:05] ROCm environment configured
[2026-01-22 17:00:10] Starting training...
[2026-01-22 17:00:15] Training process started (PID: 12345)
[2026-01-22 17:00:45] Training progress: 0 -> 100
[2026-01-22 17:01:15] Training progress: 100 -> 200
...
[2026-01-22 17:45:30] ⚠️  GPU HANG DETECTED! Initiating recovery...
[2026-01-22 17:45:35] Recovery attempt 1/5
[2026-01-22 17:45:40] Failure reason: GPU hang detected
[2026-01-22 17:45:45] Creating config for Level 0 (Conservative)
[2026-01-22 17:45:50] Resuming from checkpoint-92
[2026-01-22 17:46:00] Training process restarted (PID: 12346)
[2026-01-22 17:46:30] Training progress: 2582 -> 2590
...
```

## Advanced: Monitor GPU Health Separately

You can also run the original monitor script in parallel:

```bash
# Terminal 1: Auto-recovery
finetune/start_auto_recovery.sh

# Terminal 2: GPU health monitor
finetune/monitor_resume.sh
```

This provides detailed GPU metrics alongside auto-recovery logs.
