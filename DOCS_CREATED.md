# Documentation Created - Pipeline & Auto-Recovery

Complete documentation for running the LLM fine-tuning pipeline with auto-recovery from GPU hangs.

**Created**: January 22, 2026
**Status**: All documentation complete and ready to use

---

## 📚 New Documentation Files

### 1. **README_PIPELINE.md** (Main Entry Point)
High-level guide to running the complete pipeline.
- One-click commands for different scenarios
- Quick setup instructions
- Common use cases
- Auto-recovery overview
- File structure

**Use this first** to understand your options.

### 2. **QUICK_REFERENCE.md** (Cheat Sheet)
Fast lookup for all common commands.
- Copy-paste commands for every operation
- Training shortcuts
- Data inspection commands
- Configuration tips
- Status checking

**Use this to quickly find any command.**

### 3. **PIPELINE_STAGES.md** (Detailed Guide)
Complete walkthrough of each stage independently.
- What each stage does
- How to run each stage alone
- Configuration options per stage
- Expected outputs
- Troubleshooting by stage
- Common workflows

**Use this when running individual stages or understanding the pipeline.**

### 4. **TRAINING_AUTO_RECOVERY.md** (Training Specifics)
Complete training documentation with auto-recovery details.
- Auto-recovery how it works
- Recovery levels (0-3)
- Monitoring training progress
- Recovery logs interpretation
- Configuration & customization
- Troubleshooting GPU hangs

**Use this for all training-related questions.**

### 5. **AGENTS.md** (Updated)
Original complete reference, now with auto-recovery information.
- Full pipeline architecture
- Configuration examples
- Performance benchmarks
- Hardware optimization
- Environment setup
- Troubleshooting guide

**Use this for comprehensive technical reference.**

---

## 🎯 Quick Start Paths

### Path 1: "I Want Everything"
1. Read: **README_PIPELINE.md** (3 min)
2. Run: `python -m pipeline.runner` (2-24 hours)
3. Result: Trained model in Ollama

### Path 2: "Just Train (Data Already Ready)"
1. Read: **TRAINING_AUTO_RECOVERY.md** (5 min)
2. Run: `./finetune/start_auto_recovery.sh` (30 min - several hours)
3. Monitor: `tail -f /tmp/training_recovery/recovery_*.log`

### Path 3: "I Need a Command"
1. Search: **QUICK_REFERENCE.md**
2. Copy command
3. Run it

### Path 4: "Understand the Pipeline"
1. Read: **README_PIPELINE.md** (3 min)
2. Read: **PIPELINE_STAGES.md** (10 min)
3. See: Each stage can run independently

---

## 🔑 Key Concepts

### Auto-Recovery System
The training system automatically detects and recovers from GPU hangs:

```
GPU Hang Detected
    ↓
Kill Zombie Processes
    ↓
Reset GPU State
    ↓
Load More Conservative Config (Level 0-3)
    ↓
Resume from Checkpoint
    ↓
Continue Training
```

Recovery levels trade speed for stability:
- Level 0: ~90% speed (similar to original)
- Level 1: ~70% speed (conservative)
- Level 2: ~50% speed (very conservative)
- Level 3: ~30% speed (ultra-safe)

### Pipeline Stages
6 independent stages that can run individually or together:

1. **COLLECT** - Clone GitHub repositories
2. **EXTRACT** - Parse code and documentation
3. **COMBINE** - Merge datasets
4. **DEDUPLICATE** - Remove duplicates and low-quality examples
5. **TRAIN** - Fine-tune model with auto-recovery
6. **EXPORT** - Convert to Ollama model

---

## 📖 Documentation Map

```
README_PIPELINE.md
├── Overview of complete pipeline
├── 3 one-click starting options
├── Links to other docs
└── Quick troubleshooting

QUICK_REFERENCE.md
├── All commands copy-paste ready
├── Stage commands
├── Training commands
├── Data inspection
├── Monitoring
└── Shortcuts (for .bashrc)

PIPELINE_STAGES.md
├── Stage 1: COLLECT - Clone repos
├── Stage 2: EXTRACT - Parse code
├── Stage 3: COMBINE - Merge data
├── Stage 4: DEDUPLICATE - Clean data
├── Stage 5: TRAIN - Fine-tune (with auto-recovery)
├── Stage 6: EXPORT - Create Ollama model
├── Workflows (multi-stage combinations)
└── Troubleshooting by stage

TRAINING_AUTO_RECOVERY.md
├── Overview of auto-recovery
├── 3 ways to start training
├── Recovery levels (0-3) explained
├── Monitoring training
├── Recovery logs format
├── Configuration options
├── Troubleshooting hangs
└── Performance expectations

AGENTS.md (Updated)
├── Complete technical reference
├── All configuration options
├── Hardware optimization
├── Performance benchmarks
├── Environment setup
├── Coding practices
└── Advanced troubleshooting
```

---

## 🚀 Getting Started

### Step 1: Choose Your Path
- **Complete pipeline?** → README_PIPELINE.md
- **Just training?** → TRAINING_AUTO_RECOVERY.md
- **Need a command?** → QUICK_REFERENCE.md
- **Understanding stages?** → PIPELINE_STAGES.md

### Step 2: Run Your Command
Each document has copy-paste ready commands.

### Step 3: Monitor Progress
- Training: `tail -f /tmp/training_recovery/recovery_*.log`
- Data: Check output files in `data/training/`
- Checkpoints: `ls finetune/output/terraform-expert-v2/checkpoint-*/`

### Step 4: Use Result
After export completes:
```bash
ollama run your-model-name "your question"
```

---

## 💡 Example Scenarios

### Scenario 1: Train on New GitHub Repository
```bash
# 1. Edit config/pipeline_config.yaml
#    - Add git_source with your repo URL

# 2. Run pipeline (can skip if data exists)
python -m pipeline.runner

# 3. Monitor training
tail -f /tmp/training_recovery/recovery_*.log

# 4. Use model
ollama run your-model "your question"
```
Reference: **PIPELINE_STAGES.md** → Stage 1

### Scenario 2: GPU Hang During Training
```bash
# Automatically handled! But if needed:

# 1. Check logs
tail -f /tmp/training_recovery/recovery_*.log

# 2. Wait for auto-recovery
# The system automatically retries with more conservative settings

# 3. Manual GPU reset (if needed)
sudo rocm-smi --gpureset
```
Reference: **TRAINING_AUTO_RECOVERY.md** → Troubleshooting

### Scenario 3: Data Processing Only
```bash
python -m pipeline.runner --skip train,export
# Output: /data/training/deduped.json
```
Reference: **QUICK_REFERENCE.md** → Skip Stages

### Scenario 4: Resume Training
```bash
# Auto-recovery already handles resume from checkpoint
./finetune/start_auto_recovery.sh

# Or if checkpoint missing:
python -m pipeline.runner --stage train
```
Reference: **TRAINING_AUTO_RECOVERY.md** → Quick Start

---

## 🛠️ Core Components Created

### Training with Auto-Recovery
- `finetune/auto_recover.py` - Main recovery engine
- `finetune/start_auto_recovery.sh` - Launcher script
- `finetune/setup_auto_recovery.sh` - Systemd installer
- `finetune/training-recovery.service` - Systemd config
- `finetune/axolotl_config_v2_resume.yaml` - Conservative training config
- Recovery level configs generated dynamically

### Documentation
- `README_PIPELINE.md` - Entry point
- `QUICK_REFERENCE.md` - Command cheat sheet
- `PIPELINE_STAGES.md` - Stage-by-stage guide
- `TRAINING_AUTO_RECOVERY.md` - Training specific
- `AGENTS.md` - Updated with auto-recovery info
- `DOCS_CREATED.md` - This file

---

## 📋 File Organization

```
llm_training_web_data/
├── DOCUMENTATION
│   ├── README_PIPELINE.md              ← START HERE
│   ├── QUICK_REFERENCE.md              ← Commands
│   ├── PIPELINE_STAGES.md              ← Each stage
│   ├── TRAINING_AUTO_RECOVERY.md       ← Training details
│   ├── AGENTS.md                       ← Full reference
│   └── DOCS_CREATED.md                 ← This file
│
├── CONFIGURATION
│   └── config/pipeline_config.yaml
│
├── TRAINING (with Auto-Recovery)
│   ├── finetune/
│   │   ├── auto_recover.py             ← Recovery engine
│   │   ├── start_auto_recovery.sh      ← Start training
│   │   ├── setup_auto_recovery.sh      ← Setup systemd
│   │   ├── training-recovery.service
│   │   ├── axolotl_config_v2_resume.yaml
│   │   └── output/terraform-expert-v2/
│   │       └── checkpoint-92/          ← Resume point
│   └── /tmp/training_recovery/
│       └── recovery_*.log              ← Training logs
│
├── DATA
│   ├── repos/                          ← Cloned repos
│   └── data/training/
│       ├── combined.json               ← Stage 3
│       └── deduped.json                ← Stage 4
│
└── PIPELINE
    └── pipeline/
        ├── runner.py                   ← Main orchestrator
        ├── config_loader.py
        ├── data_extractor.py
        ├── data_processor.py
        └── model_exporter.py
```

---

## ✅ Checklist: What's Ready

- ✅ **Auto-recovery system** implemented and documented
- ✅ **Complete pipeline** documented with 6 stages
- ✅ **Training with recovery** fully explained
- ✅ **One-click commands** provided for all scenarios
- ✅ **Individual stage execution** documented
- ✅ **Monitoring & troubleshooting** guides included
- ✅ **Configuration examples** for all stages
- ✅ **Quick reference** with all common commands
- ✅ **File structure** explained
- ✅ **Recovery levels** (0-3) documented

---

## 🎓 Learning Order

**Minimum (5 min)**:
1. README_PIPELINE.md

**Standard (15 min)**:
1. README_PIPELINE.md
2. QUICK_REFERENCE.md

**Complete (30 min)**:
1. README_PIPELINE.md
2. PIPELINE_STAGES.md
3. TRAINING_AUTO_RECOVERY.md

**Expert (60 min)**:
1. All of the above
2. AGENTS.md (complete technical reference)

---

## 🔗 Quick Links

| Task | Command | Doc |
|------|---------|-----|
| Run everything | `python -m pipeline.runner` | README_PIPELINE.md |
| Train only | `./finetune/start_auto_recovery.sh` | TRAINING_AUTO_RECOVERY.md |
| Data only | `python -m pipeline.runner --skip train,export` | QUICK_REFERENCE.md |
| Export | `python -m pipeline.runner --stage export` | QUICK_REFERENCE.md |
| Run stage | `python -m pipeline.runner --stage extract` | PIPELINE_STAGES.md |
| Monitor | `tail -f /tmp/training_recovery/recovery_*.log` | QUICK_REFERENCE.md |

---

## 🚀 Next Steps

1. **Choose your path** from "Quick Start Paths" above
2. **Read the recommended document** (3-30 min)
3. **Copy a command** from QUICK_REFERENCE.md
4. **Run it** and monitor progress
5. **Troubleshoot** if needed using the relevant doc

---

## 📞 Getting Help

| Issue | Document | Section |
|-------|----------|---------|
| "Where do I start?" | README_PIPELINE.md | Quick Start |
| "How do I run X?" | QUICK_REFERENCE.md | Search by task |
| "What does stage Y do?" | PIPELINE_STAGES.md | Stage Y section |
| "GPU hang during training" | TRAINING_AUTO_RECOVERY.md | Troubleshooting |
| "Advanced configuration" | AGENTS.md | Configuration Guide |
| "Performance issue" | AGENTS.md | Hardware Optimization |

---

**Status**: ✅ All documentation complete and ready to use!

**Start here**: Read README_PIPELINE.md or QUICK_REFERENCE.md
