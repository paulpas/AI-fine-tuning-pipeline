# Training with Auto-Recovery

Automatic GPU hang detection, recovery, and training resume with progressively conservative settings.

## Overview

The auto-recovery system monitors training in real-time and automatically:
- **Detects failures**: GPU hangs (kernel exceptions), process crashes, progress stalls
- **Recovers automatically**: Kills zombies, resets GPU state, resumes from checkpoint
- **Escalates strategies**: Uses progressively conservative settings (4 recovery levels)
- **Retries intelligently**: Up to 5 recovery attempts before giving up
- **Logs everything**: Detailed recovery logs for debugging

## Quick Start

### Start Training with Auto-Recovery

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Option 1: Foreground (see logs in terminal)
./finetune/start_auto_recovery.sh

# Option 2: Background (runs in background)
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log

# Option 3: Systemd Service (persistent, survives reboot)
cd finetune
sudo ./setup_auto_recovery.sh
sudo systemctl start training-recovery
sudo systemctl enable training-recovery    # Auto-start on reboot
sudo journalctl -u training-recovery -f    # Monitor
```

## What Gets Auto-Recovered

### Detected Issues
✓ **GPU hangs** - Kernel exceptions from GPU (e.g., "GPU Hang")
✓ **Process crashes** - Training process exits unexpectedly
✓ **Progress stalls** - Training not advancing steps for 50+ seconds

### Automatic Actions
1. Kill zombie processes (cleanup)
2. Reset GPU state (with `rocm-smi --gpureset`)
3. Load more conservative training config
4. Resume from checkpoint-92 (230 MB)
5. Restart training with new config
6. Resume monitoring

## Recovery Levels

When training fails, the system automatically escalates through 4 recovery levels.

Each level trades training speed for stability:

```
Level 0 (First attempt)  → Most similar to original config
├─ sequence_len: 1536 (↓ from 2048)
├─ sample_packing: false
├─ micro_batch_size: 1
├─ gradient_accumulation: 4
└─ Expected speed: ~90% of original

Level 1 (Second attempt)  → More conservative
├─ sequence_len: 1024
├─ micro_batch_size: 1
├─ gradient_accumulation: 2
└─ Expected speed: ~70% of original

Level 2 (Third attempt)  → Very conservative
├─ sequence_len: 768
├─ micro_batch_size: 1
├─ gradient_accumulation: 1
└─ Expected speed: ~50% of original

Level 3 (Fourth+ attempts)  → Ultra conservative (single-step mode)
├─ sequence_len: 512
├─ micro_batch_size: 1
├─ gradient_accumulation: 1
└─ Expected speed: ~30% of original
```

## Monitor Recovery Progress

### View Recovery Logs (Real-time)

```bash
# Show all recovery events
tail -f /tmp/training_recovery/recovery_*.log

# Search for specific events
grep -i "hang\|error\|failure\|recovery" /tmp/training_recovery/*.log
```

### Example Recovery Log

```
[2026-01-22 17:10:15] Training Auto-Recovery System Started
[2026-01-22 17:10:20] ROCm environment configured
[2026-01-22 17:10:30] Starting training...
[2026-01-22 17:10:45] Training process started (PID: 123456)
[2026-01-22 17:10:50] Training progress: 0 -> 100
[2026-01-22 17:11:20] Training progress: 100 -> 200
...
[2026-01-22 17:45:30] Training progress: 2580 -> 2582
[2026-01-22 17:45:35] ⚠️  GPU HANG DETECTED! Initiating recovery...
[2026-01-22 17:45:40] Recovery attempt 1/5
[2026-01-22 17:45:45] Failure reason: GPU hang detected
[2026-01-22 17:45:50] Creating Level 0 (Conservative) config
[2026-01-22 17:45:55] Killing zombie processes...
[2026-01-22 17:46:00] Resetting GPU state with rocm-smi...
[2026-01-22 17:46:10] GPU reset completed
[2026-01-22 17:46:15] Resuming from checkpoint-92
[2026-01-22 17:46:25] Training process started (PID: 123457)
[2026-01-22 17:46:50] Training progress: 2582 -> 2590
...
```

### Check Training Progress

```bash
# Get latest step from training state
python3 -c "
import json
with open('/home/paulpas/git/ideas/llm_training_web_data/finetune/output/terraform-expert-v2/trainer_state.json') as f:
    state = json.load(f)
    step = state['global_step']
    total = 3495
    pct = 100 * step / total
    print(f'Progress: {step}/{total} ({pct:.1f}%)')
"

# Or check GPU status
rocm-smi
```

## Stop Training

```bash
# If running in foreground: Ctrl+C

# If running in background (nohup)
kill $(cat /tmp/training_pid.txt)

# If running as systemd service
sudo systemctl stop training-recovery
```

## Recovery Configuration

The system uses two configs:

1. **axolotl_config_v2_resume.yaml** - Initial conservative config (Level 0)
2. **axolotl_config_recovery_levelN.yaml** - Generated for each recovery attempt

### Modify Recovery Behavior

Edit `/finetune/auto_recover.py` to adjust:

```python
CONFIG = {
    'max_recovery_attempts': 5,          # Max retry attempts
    'hang_check_interval': 30,           # Check interval (seconds)
    'process_check_interval': 10,        # Monitor interval (seconds)
}

RECOVERY_LEVELS = [
    {
        'name': 'Level 0',
        'sequence_len': 1536,            # Adjust these
        'micro_batch_size': 1,
        'gradient_accumulation_steps': 4,
        'learning_rate': 0.00005,
    },
    # ... modify other levels as needed
]
```

## Troubleshooting

### Q: Is training running?
```bash
# Check process
pgrep -f auto_recover.py
# If output is PID: yes, running
# If no output: not running
```

### Q: Where are logs?
```bash
# All logs in this directory
ls -lht /tmp/training_recovery/
tail -f /tmp/training_recovery/recovery_*.log
```

### Q: Check GPU health
```bash
# Current GPU status
rocm-smi

# GPU temperatures
rocm-smi | grep "°C"

# GPU utilization
rocm-smi | grep "%"
```

### Q: Still getting GPU hangs?

1. **Check GPU physically**: Overheating? Fan issues?
2. **Manual GPU reset**:
   ```bash
   sudo rocm-smi --gpureset
   sleep 5
   ```
3. **More conservative settings**: Edit `auto_recover.py` recovery levels
4. **System reboot** (last resort):
   ```bash
   sudo reboot
   ```

### Q: How do I stop auto-recovery?

```bash
# Foreground: Ctrl+C

# Background:
pkill -f auto_recover.py

# Systemd:
sudo systemctl stop training-recovery
```

## Performance Expectations

When training resumes with progressively conservative settings:

```
Initial (Level 0):    ~1.5 steps/second  (original speed)
After 1st recovery:   ~1.3 steps/second  (90% of original)
After 2nd recovery:   ~1.0 steps/second  (70% of original)
After 3rd recovery:   ~0.8 steps/second  (50% of original)
After 4th recovery:   ~0.5 steps/second  (30% of original)
```

Progress is more important than speed - the system prioritizes completion.

## Checkpoint Information

**Current Checkpoint**: checkpoint-92 (from Jan 13, 19:17)
- Size: 230 MB
- Global step: ~2582
- Progress: ~73% of training
- Remaining steps: ~913 steps
- Estimated completion: 10-20 minutes (at Level 0) or 30-60 minutes (at Level 2-3)

## Advanced: Systemd Service

If using systemd for persistent monitoring:

### Service Management
```bash
# Start service
sudo systemctl start training-recovery

# Enable auto-start on boot
sudo systemctl enable training-recovery

# Check status
sudo systemctl status training-recovery

# View logs
sudo journalctl -u training-recovery -f

# Stop service
sudo systemctl stop training-recovery

# Restart service
sudo systemctl restart training-recovery
```

### Service Configuration
Edit `/etc/systemd/system/training-recovery.service` (requires sudo):

```ini
[Service]
# Restart settings
Restart=on-failure
RestartSec=30                    # Wait 30s before restart
StartLimitInterval=600           # In 600s window
StartLimitBurst=5                # Max 5 restarts

# Resource limits (optional)
# CPUQuota=80%
# MemoryLimit=100G
```

## Files

```
finetune/
├── auto_recover.py                    # Main recovery engine
├── start_auto_recovery.sh             # Launcher script
├── setup_auto_recovery.sh             # Systemd installer
├── training-recovery.service          # Systemd config
├── axolotl_config_v2_resume.yaml      # Level 0 config
└── output/
    └── terraform-expert-v2/
        ├── checkpoint-92/             # Resume point
        ├── trainer_state.json         # Training state
        └── ...

/tmp/training_recovery/
├── recovery_20260122_171000.log
├── recovery_20260122_172000.log       # View with: tail -f
└── ...
```

## Next Steps

1. **Monitor training**: `tail -f /tmp/training_recovery/recovery_*.log`
2. **Check GPU**: `watch rocm-smi` (updates every 2 seconds)
3. **Check progress**: `grep "Training progress" /tmp/training_recovery/*.log`
4. **Once complete**: Run export stage to convert to Ollama model

## Integration with Full Pipeline

See `PIPELINE_STAGES.md` for how training fits into the complete pipeline:

```
STAGE 1: COLLECT     → Clone repositories
STAGE 2: EXTRACT     → Parse code & docs
STAGE 3: COMBINE     → Merge datasets
STAGE 4: DEDUPLICATE → Remove duplicates
STAGE 5: TRAIN       → Fine-tune with auto-recovery ← YOU ARE HERE
STAGE 6: EXPORT      → Convert to Ollama model
```

---

For complete pipeline documentation, see `AGENTS.md` and `PIPELINE_STAGES.md`.
