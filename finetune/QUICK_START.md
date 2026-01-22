# Auto-Recovery: Quick Start

## Start Training with Auto-Recovery Now

### Option 1: Simple (Foreground)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
./finetune/start_auto_recovery.sh
```

Press `Ctrl+C` to stop.

### Option 2: Background (nohup)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
echo $! > /tmp/training_pid.txt

# Monitor logs
tail -f /tmp/auto_recovery.log
```

### Option 3: Systemd Service (Persistent)

**First-time setup:**
```bash
cd /home/paulpas/git/ideas/llm_training_web_data/finetune
sudo ./setup_auto_recovery.sh
```

**Start the service:**
```bash
sudo systemctl start training-recovery
sudo systemctl enable training-recovery
```

**Monitor:**
```bash
sudo journalctl -u training-recovery -f
```

## What Gets Auto-Recovered

If training encounters:
- ✓ GPU hangs (kernel exceptions)
- ✓ Process crashes
- ✓ Progress stalls (no step advancement)

The system will:
1. Kill zombie processes
2. Reset GPU state
3. Resume from checkpoint with more conservative settings
4. Automatically retry (up to 5 times)

## Key Files

| File | Purpose |
|------|---------|
| `auto_recover.py` | Core auto-recovery engine |
| `start_auto_recovery.sh` | Wrapper script to start recovery |
| `setup_auto_recovery.sh` | Install systemd service |
| `training-recovery.service` | Systemd service definition |
| `axolotl_config_v2_resume.yaml` | Conservative training config |
| `AUTO_RECOVERY.md` | Detailed documentation |

## Monitor Recovery Progress

**While running:**
```bash
# View recovery logs
tail -f /tmp/training_recovery/recovery_*.log

# Check GPU status
rocm-smi

# View all current recovery processes
ps aux | grep auto_recover

# Systemd logs (if using service)
sudo journalctl -u training-recovery -f
```

## Stop Training

```bash
# If running in foreground: Ctrl+C

# If running with nohup
pkill -f auto_recover.py

# If using systemd
sudo systemctl stop training-recovery
```

## Troubleshooting

**Check if auto-recovery is running:**
```bash
pgrep -f auto_recover.py
# If output is blank, it's not running
```

**View latest recovery logs:**
```bash
ls -lt /tmp/training_recovery/*.log | head -1 | awk '{print $NF}' | xargs tail -50
```

**Check systemd service status:**
```bash
sudo systemctl status training-recovery
```

## Expected Recovery Sequence

When GPU hang occurs:

```
[INFO] Training progress: 2582 -> 2583
[WARNING] GPU hang detected!
[ERROR] Recovery attempt 1/5
[INFO] Killing zombie processes...
[INFO] Resetting GPU state...
[INFO] Creating Level 0 (Conservative) config
[INFO] Resuming from checkpoint-92
[INFO] Training process restarted (PID: xxxxx)
[INFO] Training progress: 2583 -> 2584
...
```

## Next Steps

- See `AUTO_RECOVERY.md` for detailed documentation
- Monitor `/tmp/training_recovery/` for recovery logs
- Use `rocm-smi` to check GPU health
- Adjust recovery levels if needed (see AUTO_RECOVERY.md)

---

**Status:** Training auto-recovery is ready to use! 🚀
