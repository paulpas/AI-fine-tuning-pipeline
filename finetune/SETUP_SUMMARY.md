# Auto-Recovery Setup Summary

## What Was Created

A complete auto-recovery system for your GPU-hung training pipeline.

### Core Components

#### 1. **auto_recover.py** (14 KB)
   - Main auto-recovery engine
   - Monitors GPU health, training progress, process status
   - Detects hangs, crashes, and stalls
   - Auto-recovers with progressively conservative settings
   - Allows up to 5 recovery attempts

#### 2. **start_auto_recovery.sh** (1.6 KB)
   - User-friendly wrapper to start recovery
   - Checks prerequisites
   - Activates venv and launches auto-recovery
   - Can run in foreground or background

#### 3. **setup_auto_recovery.sh** (1.6 KB)
   - Installs systemd service for persistent monitoring
   - Run once with sudo to enable service mode
   - Allows training to restart on system reboot

#### 4. **training-recovery.service**
   - Systemd service definition
   - Manages training as a system service
   - Auto-restarts on failure (with exponential backoff)
   - Logs to journal

#### 5. **axolotl_config_v2_resume.yaml** (1.8 KB)
   - Conservative training config for initial recovery
   - Uses reduced sequence length (1536 instead of 2048)
   - Disabled sample packing (memory saver)
   - Reduced batch size and gradient accumulation

### Documentation

#### 6. **AUTO_RECOVERY.md** (Comprehensive)
   - Detailed operation of auto-recovery system
   - Recovery levels (0-3) with settings table
   - How to monitor and troubleshoot
   - Configuration options

#### 7. **QUICK_START.md**
   - Quick launch guide
   - File reference
   - Common commands
   - Expected behavior

#### 8. **SETUP_SUMMARY.md** (This file)
   - Overview of what was created
   - System readiness checklist
   - Launch instructions

## System Readiness Checklist

✓ Python dependencies available (psutil, subprocess, json, etc.)
✓ Resume config created: axolotl_config_v2_resume.yaml
✓ Checkpoint available: checkpoint-92 (230 MB)
✓ ROCm drivers ready (4x MI50 GPUs detected)
✓ Venv activated and ready to use
✓ All scripts executable and permissions set

## How Auto-Recovery Works

```
┌─────────────────────────────────────────┐
│   Start Training with auto_recover.py   │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  Monitor Training (every 10 seconds)    │
│  ✓ GPU hangs?                           │
│  ✓ Process alive?                       │
│  ✓ Progress advancing?                  │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
        YES          NO (Failure Detected)
         │           │
         ↓           ↓
    Continue    ┌────────────────────────┐
                │ Initiate Recovery:     │
                │ 1. Kill zombies        │
                │ 2. Reset GPU           │
                │ 3. Load conservative   │
                │    config (Level 0-3)  │
                │ 4. Resume from         │
                │    checkpoint-92       │
                └────────────────┬───────┘
                                 │
                        ┌────────┴────────┐
                        │                 │
                    Attempt < 5       Attempt >= 5
                        │                 │
                        ↓                 ↓
                  Resume Training    ❌ Exit with Error
                        │
                        └──→ Back to monitoring
```

## Quick Launch Options

### Option 1: Foreground (Simple, for testing)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
./finetune/start_auto_recovery.sh

# Logs appear in stdout
# Press Ctrl+C to stop
```

**Time to start:** ~30 seconds
**Logs location:** Stdout + /tmp/training_recovery/

---

### Option 2: Background (nohup, for overnight runs)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
echo $! > /tmp/training_pid.txt

# Monitor
tail -f /tmp/auto_recovery.log
tail -f /tmp/training_recovery/recovery_*.log

# Stop
kill $(cat /tmp/training_pid.txt)
```

**Time to start:** ~30 seconds
**Logs location:** /tmp/auto_recovery.log + /tmp/training_recovery/
**Persistence:** Lost on reboot

---

### Option 3: Systemd Service (Recommended, for production)
```bash
# One-time setup
cd /home/paulpas/git/ideas/llm_training_web_data/finetune
sudo ./setup_auto_recovery.sh

# Start service
sudo systemctl start training-recovery

# Enable auto-start on reboot
sudo systemctl enable training-recovery

# Monitor
sudo journalctl -u training-recovery -f

# Stop
sudo systemctl stop training-recovery
```

**Time to start:** ~2 minutes (setup first time)
**Logs location:** systemd journal + /tmp/training_recovery/
**Persistence:** Survives reboots and system restarts

---

## Recovery Levels Explained

When training fails, system attempts recovery with increasingly conservative settings:

```
Level 0 (First attempt)
├─ sequence_len: 1536 (↓ from 2048)
├─ sample_packing: false
├─ micro_batch_size: 1
├─ gradient_accumulation: 4
└─ Speed: ~90% of original

Level 1 (Second attempt)
├─ sequence_len: 1024
├─ sample_packing: false
├─ micro_batch_size: 1
├─ gradient_accumulation: 2
└─ Speed: ~70% of original

Level 2 (Third attempt)
├─ sequence_len: 768
├─ sample_packing: false
├─ micro_batch_size: 1
├─ gradient_accumulation: 1
└─ Speed: ~50% of original

Level 3 (Fourth+ attempts)
├─ sequence_len: 512
├─ sample_packing: false
├─ micro_batch_size: 1
├─ gradient_accumulation: 1
└─ Speed: ~30% of original
```

Each level trades speed for stability to avoid future hangs.

## Monitoring Auto-Recovery

### Real-time Monitoring
```bash
# Option 1: Direct log tail
tail -f /tmp/training_recovery/recovery_*.log

# Option 2: Systemd journal (if using service)
sudo journalctl -u training-recovery -f

# Option 3: GPU monitoring
watch rocm-smi  # Refreshes every 2 seconds

# Option 4: Check process is running
ps aux | grep auto_recover
```

### Example Log Output
```
[2026-01-22 17:10:00] Starting auto-recovery training...
[2026-01-22 17:10:05] ROCm environment configured
[2026-01-22 17:10:15] Training process started (PID: 123456)
[2026-01-22 17:10:45] Training progress: 0 -> 100
[2026-01-22 17:11:15] Training progress: 100 -> 200
...
[2026-01-22 17:45:00] Training progress: 2580 -> 2582
[2026-01-22 17:45:30] ⚠️  WARNING: GPU hang detected!
[2026-01-22 17:45:35] ERROR: Training failure detected: GPU hang detected
[2026-01-22 17:45:40] Recovery attempt 1/5
[2026-01-22 17:45:45] Failure reason: GPU hang detected
[2026-01-22 17:45:50] Creating config for Level 0 (Conservative)
[2026-01-22 17:46:00] Cleaned zombie processes
[2026-01-22 17:46:05] Attempting GPU reset...
[2026-01-22 17:46:15] GPU reset completed
[2026-01-22 17:46:20] Starting training with recovery config...
[2026-01-22 17:46:30] Training process started (PID: 123457)
[2026-01-22 17:47:00] Training progress: 2582 -> 2585
[2026-01-22 17:47:30] Training progress: 2585 -> 2590
...
```

## Troubleshooting

### Q: Is auto-recovery running?
```bash
pgrep -f auto_recover.py
# If no output, it's not running
# If output (PID), it is running
```

### Q: Where are logs?
```bash
ls /tmp/training_recovery/
# View latest: tail -f /tmp/training_recovery/recovery_*.log
```

### Q: How do I stop it?
```bash
# Foreground: Ctrl+C

# Background (nohup): kill $PID
kill $(pgrep -f auto_recover.py)

# Systemd: sudo systemctl stop training-recovery
```

### Q: What if it keeps failing?
Check `/tmp/training_recovery/` logs for specific error.
May need to:
1. Manually check GPU health: `rocm-smi`
2. Adjust recovery levels (see AUTO_RECOVERY.md)
3. Reboot system to reset GPU state

## File Structure

```
finetune/
├── auto_recover.py                    # Main engine
├── start_auto_recovery.sh             # Launcher
├── setup_auto_recovery.sh             # Systemd setup
├── training-recovery.service          # Systemd config
├── axolotl_config_v2_resume.yaml      # Conservative config
├── AUTO_RECOVERY.md                   # Full documentation
├── QUICK_START.md                     # Quick reference
├── SETUP_SUMMARY.md                   # This file
└── output/
    └── terraform-expert-v2/
        ├── checkpoint-92/             # Resume checkpoint
        ├── trainer_state.json         # Training state
        └── ...
```

## Next Steps

### To Start Training Now:
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
./finetune/start_auto_recovery.sh
```

### To Set Up as Systemd Service:
```bash
cd /home/paulpas/git/ideas/llm_training_web_data/finetune
sudo ./setup_auto_recovery.sh
sudo systemctl start training-recovery
sudo systemctl enable training-recovery
```

### To Monitor Progress:
```bash
tail -f /tmp/training_recovery/recovery_*.log
```

## Performance Expectations

- **Initial speed:** ~1.5 iterations/second (with Level 0 config)
- **After Level 1 recovery:** ~1.3 iterations/second
- **After Level 2 recovery:** ~1.0 iterations/second
- **After Level 3 recovery:** ~0.5 iterations/second

Progress is more important than speed - the system prioritizes stability.

## Questions?

See:
- `AUTO_RECOVERY.md` - Detailed operation and configuration
- `QUICK_START.md` - Common commands and status checks
- Recovery logs: `/tmp/training_recovery/recovery_*.log`

---

**Status:** ✅ Auto-recovery system is ready to use!

You can start training now with automatic GPU hang recovery.
