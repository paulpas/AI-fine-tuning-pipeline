# LLM Fine-Tuning Pipeline

One-click to fine-tune LLMs on your code, with auto-recovery from GPU hangs. Complete end-to-end: data collection → code extraction → training → export.

## 🚀 Start Here

Choose what you want to do:

### Option 1: Run Everything (Complete Pipeline)
```bash
python -m pipeline.runner
```
Automatically: Collects repos → Extracts code → Combines data → Deduplicates → Trains → Exports to Ollama

**Time**: 2-24 hours (depends on data size and hardware)

### Option 2: Training Only (with Auto-Recovery)
```bash
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log
```
Assumes data is already prepared. Includes automatic GPU hang recovery.

**Time**: 30 minutes to several hours

### Option 3: Run Individual Stages
```bash
python -m pipeline.runner --stage extract,combine,dedupe,train,export
```
Choose which stages to run. Skip others.

---

## 📚 Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **QUICK_REFERENCE.md** | Copy-paste commands for everything | 5 min |
| **PIPELINE_STAGES.md** | Detailed guide to each stage | 10 min |
| **TRAINING_AUTO_RECOVERY.md** | Training with auto-recovery details | 10 min |
| **AGENTS.md** | Complete technical reference | 30 min |

**→ Start with QUICK_REFERENCE.md**

---

## 🎯 Common Scenarios

### Scenario 1: "I want to train on my own code"
```bash
# 1. Edit config/pipeline_config.yaml
#    - Add your GitHub repo to git_sources
#    - Set base_model

# 2. Run pipeline
python -m pipeline.runner

# 3. Use the model
ollama run your-model "your question"
```
See: PIPELINE_STAGES.md → Stage 1

### Scenario 2: "Training failed, resume with auto-recovery"
```bash
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log
```
Auto-recovery automatically:
- Detects GPU hangs/crashes
- Resumes from last checkpoint
- Uses more conservative settings if needed
- Retries up to 5 times

See: TRAINING_AUTO_RECOVERY.md

### Scenario 3: "I have data, just need to train"
```bash
python -m pipeline.runner --skip collect,extract,combine,dedupe
```
Or with auto-recovery:
```bash
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
```
See: QUICK_REFERENCE.md → Training

### Scenario 4: "Export trained model to Ollama"
```bash
python -m pipeline.runner --stage export
```
See: PIPELINE_STAGES.md → Stage 6

### Scenario 5: "Just process data, no training"
```bash
python -m pipeline.runner --skip train,export
```
Output: `/data/training/deduped.json`

---

## 🔧 Setup

### Requirements
- Python 3.10+
- 4x AMD/NVIDIA GPUs (or 1x GPU with reduced batch size)
- 100+ GB disk space

### One-time Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## 📊 Pipeline Architecture

```
DATA COLLECTION        DATA PROCESSING        TRAINING & EXPORT
    ↓                      ↓                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                    6-STAGE PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Stage 1: COLLECT           Clone GitHub repositories             │
│   ↓                                                               │
│ Stage 2: EXTRACT           Parse Python code & docs              │
│   ↓                                                               │
│ Stage 3: COMBINE           Merge all datasets                    │
│   ↓                                                               │
│ Stage 4: DEDUPLICATE       Remove duplicates & low quality       │
│   ↓                                                               │
│ Stage 5: TRAIN             Fine-tune with auto-recovery          │
│   ↓                        (GPU hang detection & resume)          │
│ Stage 6: EXPORT            Convert to Ollama model               │
│   ↓                                                               │
│ OUTPUT: Ollama model ready for inference                          │
└─────────────────────────────────────────────────────────────────┘
```

**Each stage** can run independently or skip stages as needed.

---

## 🛡️ Auto-Recovery (GPU Hang Detection)

**The pipeline includes automatic GPU hang detection and recovery.**

When training encounters GPU hangs:
1. **Detects** the hang via kernel logs
2. **Stops** the training process
3. **Resets** GPU state
4. **Resumes** from checkpoint with more conservative settings
5. **Retries** (up to 5 times)

Example recovery sequence:
```
[INFO] Training progress: 2580 -> 2582
[WARNING] GPU hang detected!
[INFO] Recovery attempt 1/5
[INFO] Resetting GPU state...
[INFO] Using Level 0 (Conservative) config
[INFO] Resuming from checkpoint-92
[INFO] Training process restarted
[INFO] Training progress: 2582 -> 2590
```

Recovery levels get progressively more conservative:
- **Level 0**: ~90% speed
- **Level 1**: ~70% speed
- **Level 2**: ~50% speed
- **Level 3**: ~30% speed (ultra-safe)

See: TRAINING_AUTO_RECOVERY.md for complete details.

---

## 📝 Configuration

### Minimal Configuration
Edit `config/pipeline_config.yaml`:

```yaml
pipeline:
  name: my-model

git_sources:
  - name: my-repo
    url: https://github.com/user/repo
    enabled: true

training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  num_gpus: 4
  num_epochs: 3
```

### Full Configuration
See AGENTS.md → Configuration Guide for all options:
- LoRA settings
- Hyperparameters
- Data processing
- Export settings
- Profiles (test/production)

---

## 📊 Run Complete Pipeline

### One Command
```bash
python -m pipeline.runner
```

### With Custom Config
```bash
python -m pipeline.runner --config config/pipeline_config.yaml
```

### Using Profile
```bash
python -m pipeline.runner --profile production  # Full quality
python -m pipeline.runner --profile test        # Quick test
```

### Override Settings
```bash
export PIPELINE_TRAINING_NUM_GPUS=2
export PIPELINE_TRAINING_BASE_MODEL="mistral-ai/Mistral-7B"
python -m pipeline.runner
```

---

## 🏃 Running Stages Independently

See PIPELINE_STAGES.md for detailed stage documentation.

### Stage Commands
```bash
# Collect repos
python -m pipeline.runner --stage collect

# Extract code
python -m pipeline.runner --stage extract

# Combine datasets
python -m pipeline.runner --stage combine

# Deduplicate
python -m pipeline.runner --stage dedupe

# Train (with auto-recovery)
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &

# Export
python -m pipeline.runner --stage export
```

### Skip Stages
```bash
# Skip training and export
python -m pipeline.runner --skip train,export

# Skip data collection (use existing data)
python -m pipeline.runner --skip collect,extract

# Data processing only
python -m pipeline.runner --skip train,export
```

---

## 🔍 Monitor Training

### View Progress
```bash
# Auto-recovery logs
tail -f /tmp/training_recovery/recovery_*.log

# Or check step count
python3 -c "
import json
with open('finetune/output/terraform-expert-v2/trainer_state.json') as f:
    state = json.load(f)
    step = state['global_step']
    total = 3495
    print(f'Progress: {step}/{total} ({100*step/total:.1f}%)')
"

# GPU status
rocm-smi
```

### If Using Systemd Service
```bash
sudo journalctl -u training-recovery -f
```

---

## 🤖 Use Trained Model

After export completes:

```bash
# Run the model
ollama run your-model-name "Your question"

# Examples
ollama run python-expert "How do I use async/await?"
ollama run terraform-expert "What is a data source?"
```

---

## 🚨 Troubleshooting

### GPU Hang During Training
Auto-recovery handles this automatically! Check logs:
```bash
tail -f /tmp/training_recovery/recovery_*.log
```

### Out of Memory
Reduce batch size or sequence length in config:
```yaml
training:
  hyperparameters:
    micro_batch_size: 1    # Reduce
    sequence_len: 512      # Reduce
    gradient_accumulation_steps: 16  # Increase
```

### Training Quality Issues
See AGENTS.md → Troubleshooting section

### Export/GGUF Conversion Fails
```bash
# Ensure dependencies
pip install llama-cpp-python
pip install torch torchvision

# Try again
python -m pipeline.runner --stage export
```

---

## 📂 File Structure

```
llm_training_web_data/
├── README_PIPELINE.md                 ← You are here
├── QUICK_REFERENCE.md                 ← Common commands
├── PIPELINE_STAGES.md                 ← Each stage explained
├── TRAINING_AUTO_RECOVERY.md          ← Auto-recovery details
├── AGENTS.md                          ← Complete reference
│
├── config/
│   └── pipeline_config.yaml           ← Edit this
│
├── data/
│   └── training/
│       ├── combined.json              ← Stage 3 output
│       └── deduped.json               ← Stage 4 output
│
├── finetune/
│   ├── start_auto_recovery.sh         ← Start training (with recovery)
│   ├── auto_recover.py                ← Recovery engine
│   ├── axolotl_config_v2_resume.yaml  ← Training config
│   └── output/
│       └── terraform-expert-v2/
│           ├── checkpoint-92/         ← Resume point
│           └── trainer_state.json
│
└── pipeline/
    ├── runner.py                      ← Main orchestrator
    ├── config_loader.py               ← Config management
    ├── data_extractor.py              ← Code extraction
    ├── data_processor.py              ← Dedup & combine
    └── model_exporter.py              ← Export to Ollama
```

---

## 🎓 Learning Path

1. **First time?** → Read QUICK_REFERENCE.md (5 min)
2. **Understanding stages** → Read PIPELINE_STAGES.md (10 min)
3. **Training issues** → Read TRAINING_AUTO_RECOVERY.md (10 min)
4. **Advanced config** → Read AGENTS.md (30 min)

---

## 🔗 Quick Links

- **Run everything**: `python -m pipeline.runner`
- **Train with auto-recovery**: `./finetune/start_auto_recovery.sh`
- **Monitor training**: `tail -f /tmp/training_recovery/recovery_*.log`
- **Run specific stage**: `python -m pipeline.runner --stage extract`
- **Use model**: `ollama run your-model-name`

---

## 📖 Full Documentation

See `AGENTS.md` for complete technical reference with:
- Hardware optimization
- Memory tuning
- Supported models
- Configuration examples
- Performance benchmarks
- Environment setup
- Advanced debugging

---

**Status**: ✅ Pipeline ready to run!

Start with `python -m pipeline.runner` or see QUICK_REFERENCE.md for other options.
