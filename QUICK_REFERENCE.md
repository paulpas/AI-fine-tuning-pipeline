# Quick Reference - Common Commands

Fast lookup guide for all common pipeline operations.

## One-Click Commands

### Complete Pipeline (All Stages)
```bash
python -m pipeline.runner
```
Runs: Collect → Extract → Combine → Deduplicate → Train → Export

### Training Only (with Auto-Recovery)
```bash
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log
```

### Export Only (after training)
```bash
python -m pipeline.runner --stage export
```

---

## Individual Stages

### Stage 1: Collect Repositories
```bash
python -m pipeline.runner --stage collect
```

### Stage 2: Extract Code & Docs
```bash
python -m pipeline.runner --stage extract
```

### Stage 3: Combine Datasets
```bash
python -m pipeline.runner --stage combine
```

### Stage 4: Deduplicate
```bash
python -m pipeline.runner --stage dedupe
```

### Stage 5: Train (Auto-Recovery)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log
```

### Stage 6: Export Model
```bash
python -m pipeline.runner --stage export
```

---

## Skip Stages

### Skip Training & Export
```bash
python -m pipeline.runner --skip train,export
```

### Skip Data Collection
```bash
python -m pipeline.runner --skip collect,extract
```

### Data Processing Only
```bash
python -m pipeline.runner --skip train,export
```

---

## Training with Auto-Recovery

### Start Training (Background)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
echo $! > /tmp/training_pid.txt
```

### Start Training (Systemd - Persistent)
```bash
cd /home/paulpas/git/ideas/llm_training_web_data/finetune
sudo ./setup_auto_recovery.sh
sudo systemctl start training-recovery
sudo systemctl enable training-recovery  # Auto-start on reboot
```

### Monitor Training
```bash
# View recovery logs
tail -f /tmp/training_recovery/recovery_*.log

# View systemd logs
sudo journalctl -u training-recovery -f

# View background process logs
tail -f /tmp/auto_recovery.log
```

### Check Training Progress
```bash
python3 -c "
import json
with open('/home/paulpas/git/ideas/llm_training_web_data/finetune/output/terraform-expert-v2/trainer_state.json') as f:
    state = json.load(f)
    step = state['global_step']
    total = 3495
    pct = 100 * step / total
    print(f'Progress: {step}/{total} ({pct:.1f}%)')
"
```

### Check GPU Status
```bash
rocm-smi
watch -n 1 rocm-smi    # Updates every second
```

### Stop Training
```bash
# If running in background (nohup)
kill $(cat /tmp/training_pid.txt)

# If running with systemd
sudo systemctl stop training-recovery

# Force kill if stuck
pkill -9 auto_recover
```

---

## Model Export & Usage

### Export to Ollama
```bash
python -m pipeline.runner --stage export
```

### List Available Models
```bash
ollama list
```

### Run Model
```bash
ollama run terraform-expert "Your question here"
```

### Remove Model
```bash
ollama rm terraform-expert
```

---

## Data Inspection

### View Combined Dataset
```bash
head -5 data/training/combined.json | python -m json.tool
```

### Count Examples (Combined)
```bash
python3 -c "
import json
with open('data/training/combined.json') as f:
    data = json.load(f)
    print(f'Combined: {len(data)} examples')
"
```

### Count Examples (Deduped)
```bash
python3 -c "
import json
with open('data/training/deduped.json') as f:
    data = json.load(f)
    print(f'Deduped: {len(data)} examples')
"
```

### Compare Before/After Deduplication
```bash
echo "Combined: $(wc -l < data/training/combined.json) examples"
echo "Deduped:  $(wc -l < data/training/deduped.json) examples"
python3 -c "
import json
with open('data/training/combined.json') as f:
    combined = len(json.load(f))
with open('data/training/deduped.json') as f:
    deduped = len(json.load(f))
removed = combined - deduped
pct = 100 * removed / combined
print(f'Removed: {removed} examples ({pct:.1f}%)')
"
```

### View Sample Training Data
```bash
python3 << 'EOF'
import json
with open('data/training/deduped.json') as f:
    examples = json.load(f)
    for i, ex in enumerate(examples[:3], 1):
        print(f"\n=== Example {i} ===")
        print(f"Input: {ex.get('input', 'N/A')[:100]}...")
        print(f"Output: {ex.get('output', 'N/A')[:100]}...")
EOF
```

---

## Configuration

### View Current Config
```bash
cat config/pipeline_config.yaml
```

### Edit Config
```bash
nano config/pipeline_config.yaml
```

### Validate Config Syntax
```bash
python3 -c "
import yaml
with open('config/pipeline_config.yaml') as f:
    config = yaml.safe_load(f)
    print('✓ Config is valid')
"
```

### Override with Environment Variables
```bash
export PIPELINE_TRAINING_NUM_GPUS=2
export PIPELINE_TRAINING_BASE_MODEL="mistral-ai/Mistral-7B"
python -m pipeline.runner
```

---

## Logging & Debugging

### View Training Log (After Training)
```bash
tail -100 /home/paulpas/git/ideas/llm_training_web_data/finetune/output/terraform-expert-v2/training.log
```

### View Recovery Logs
```bash
ls -lt /tmp/training_recovery/recovery_*.log | head -1 | awk '{print $NF}' | xargs tail -50
```

### Search for Errors
```bash
grep -i "error\|failed\|hang" /tmp/training_recovery/*.log
```

### Check GPU Errors
```bash
dmesg | grep -i "gpu\|hang" | tail -20
```

---

## Environment Setup

### Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify GPU Detection
```bash
python3 -c "
import torch
print(f'PyTorch GPUs: {torch.cuda.device_count()}')
"

rocm-smi --showproductname    # AMD
nvidia-smi                     # NVIDIA
```

### Set ROCm Environment (AMD)
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
```

---

## Profiles

### Test Profile (Quick)
```bash
python -m pipeline.runner --profile test
```
Faster iteration: fewer epochs, smaller batch size

### Production Profile
```bash
python -m pipeline.runner --profile production
```
Best quality: more epochs, larger LoRA rank

---

## Checkpoints & Recovery

### List Checkpoints
```bash
ls -lt finetune/output/terraform-expert-v2/checkpoint-*/
```

### Get Latest Checkpoint Number
```bash
ls finetune/output/terraform-expert-v2/ | grep checkpoint | sort -V | tail -1
```

### Check Checkpoint Size
```bash
du -sh finetune/output/terraform-expert-v2/checkpoint-*/
```

### Get Training Step from Checkpoint
```bash
python3 -c "
import json
import glob
checkpoints = glob.glob('finetune/output/terraform-expert-v2/checkpoint-*/trainer_state.json')
for ckpt in sorted(checkpoints)[-1:]:
    with open(ckpt) as f:
        state = json.load(f)
        step = state['global_step']
        print(f'{ckpt.split(\"/\")[-2]}: step {step}')
"
```

---

## System Status

### Check Disk Space
```bash
df -h
```

### Check RAM Usage
```bash
free -h
```

### Check CPU Usage
```bash
top -bn1 | head -20
```

### Check Running Processes
```bash
ps aux | grep -E "auto_recover|axolotl|train"
```

---

## File Locations

| Item | Location |
|------|----------|
| Config | `config/pipeline_config.yaml` |
| Combined Data | `data/training/combined.json` |
| Deduped Data | `data/training/deduped.json` |
| Checkpoints | `finetune/output/terraform-expert-v2/checkpoint-*/` |
| Training Logs | `finetune/output/terraform-expert-v2/training.log` |
| Recovery Logs | `/tmp/training_recovery/recovery_*.log` |
| Training State | `finetune/output/terraform-expert-v2/trainer_state.json` |

---

## Common Workflows

### Workflow 1: Quick Test
```bash
# Run on small dataset
python -m pipeline.runner --profile test
```

### Workflow 2: Full Pipeline
```bash
# Run all stages end-to-end
python -m pipeline.runner
```

### Workflow 3: Data → Train → Export
```bash
# Skip existing repos
python -m pipeline.runner --skip collect
```

### Workflow 4: Resume Training
```bash
# Auto-recovery handles checkpoints
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log
```

### Workflow 5: Add New Repository
```bash
# 1. Edit config/pipeline_config.yaml - add git_source
# 2. Extract new repo only
python -m pipeline.runner --skip collect,combine,dedupe,train,export

# 3. Recombine and train
python -m pipeline.runner --skip collect,train,export
python -m pipeline.runner --skip collect,extract,export
```

---

## Shortcuts (Add to .bashrc)

```bash
# Training shortcuts
alias train-start='cd /home/paulpas/git/ideas/llm_training_web_data && nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &'
alias train-logs='tail -f /tmp/training_recovery/recovery_*.log'
alias train-stop='pkill -f auto_recover'
alias train-status='ps aux | grep auto_recover'

# GPU shortcuts
alias gpu-status='watch -n 1 rocm-smi'
alias gpu-hot='rocm-smi | grep -E "°C|GPU"'

# Pipeline shortcuts
alias pipeline-all='python -m pipeline.runner'
alias pipeline-data='python -m pipeline.runner --skip train,export'
alias pipeline-export='python -m pipeline.runner --stage export'
```

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| **AGENTS.md** | Complete pipeline reference |
| **PIPELINE_STAGES.md** | Detailed stage-by-stage guide |
| **TRAINING_AUTO_RECOVERY.md** | Training with auto-recovery |
| **QUICK_REFERENCE.md** | This file - common commands |

---

For detailed documentation, see the relevant guide above.
