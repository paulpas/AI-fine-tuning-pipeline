# Pipeline Stages - Detailed Guide

Complete reference for running each pipeline stage independently or in sequence.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           LLM FINE-TUNING PIPELINE (6 STAGES)                   │
└─────────────────────────────────────────────────────────────────┘

STAGE 1: COLLECT
  Clone repositories from GitHub
  └─ Output: /repos/{repo_name}/

STAGE 2: EXTRACT
  Parse Python code (AST), extract docs
  └─ Output: /data/extracted/{source_name}.json

STAGE 3: COMBINE
  Merge all extracted datasets
  └─ Output: /data/training/combined.json

STAGE 4: DEDUPLICATE
  Remove duplicates & low-quality examples
  └─ Output: /data/training/deduped.json

STAGE 5: TRAIN (with Auto-Recovery)
  Fine-tune model with Axolotl
  └─ Output: /finetune/output/{model_name}/checkpoint-*/

STAGE 6: EXPORT
  Merge LoRA adapter, convert GGUF, create Ollama model
  └─ Output: Ollama model ready for inference
```

## Stage 1: COLLECT

**Purpose**: Clone GitHub repositories for training data

**Input**: Repository URLs from config
**Output**: `/repos/{repo_name}/` directories with full repo contents
**Duration**: 2-5 minutes per repo (depends on size)

### Run Stage 1 Independently

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Collect only
python -m pipeline.runner --stage collect

# Or skip everything else
python -m pipeline.runner --skip extract,combine,dedupe,train,export
```

### Configure Sources

Edit `config/pipeline_config.yaml`:

```yaml
git_sources:
  - name: k8s-python
    url: https://github.com/kubernetes-client/python
    subdirs: [examples, docs]
    type: python
    enabled: true

  - name: pytest
    url: https://github.com/pytest-dev/pytest
    subdirs: [doc/en]
    type: mixed
    enabled: true
```

### Environment Override

```bash
# Force re-download (skip cache)
export PIPELINE_COLLECTION_FORCE_FRESH=1
python -m pipeline.runner --stage collect
```

### Check Output

```bash
# List cloned repos
ls -la repos/

# Check specific repo
ls repos/k8s-python/examples/
```

---

## Stage 2: EXTRACT

**Purpose**: Parse code and documentation to create training examples

**Input**: Cloned repositories from Stage 1
**Output**: `/data/extracted/{source_name}.json` with training examples
**Duration**: 5-15 minutes depending on repo size
**Format**: Alpaca format (instruction, input, output)

### Run Stage 2 Independently

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Extract only
python -m pipeline.runner --stage extract

# Or with stage skipping
python -m pipeline.runner --skip collect,combine,dedupe,train,export
```

### Configure Extraction

Edit `config/pipeline_config.yaml`:

```yaml
processing:
  min_code_length: 50          # Minimum code snippet length
  max_code_length: 10000       # Maximum code snippet length
  output_format: alpaca        # Training format

  extraction:
    include_docstrings: true   # Extract function docstrings
    include_comments: true     # Include code comments
    include_tests: false       # Include test code
```

### Check Output

```bash
# List extracted datasets
ls -lh data/extracted/

# View sample training examples
head -5 data/extracted/k8s_extracted.json | python -m json.tool

# Count examples
python3 -c "
import json
with open('data/extracted/k8s_extracted.json') as f:
    data = json.load(f)
    print(f'Examples: {len(data)}')
"
```

---

## Stage 3: COMBINE

**Purpose**: Merge all extracted datasets into single training file

**Input**: Multiple `/data/extracted/*.json` files from Stage 2
**Output**: `/data/training/combined.json` (merged dataset)
**Duration**: < 1 minute
**Details**: Simple concatenation + metadata

### Run Stage 3 Independently

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Combine only
python -m pipeline.runner --stage combine

# Or with skipping
python -m pipeline.runner --skip collect,extract,dedupe,train,export
```

### Configure Combining

```yaml
processing:
  combine:
    preserve_source: true    # Keep source field in output
    shuffle: false           # Don't shuffle
```

### Check Output

```bash
# View combined dataset
wc -l data/training/combined.json          # Line count
python3 -c "
import json
with open('data/training/combined.json') as f:
    data = json.load(f)
    print(f'Total examples: {len(data)}')
    print(f'Sources: {set(d.get(\"source\", \"unknown\") for d in data)}')
"
```

---

## Stage 4: DEDUPLICATE

**Purpose**: Remove duplicates and low-quality examples

**Input**: `/data/training/combined.json` from Stage 3
**Output**: `/data/training/deduped.json` (cleaned dataset)
**Duration**: 5-10 minutes depending on dataset size
**Details**:
- Exact duplicate removal
- Semantic similarity detection
- Length-based filtering
- Quality scoring

### Run Stage 4 Independently

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Deduplicate only
python -m pipeline.runner --stage dedupe

# Or with skipping
python -m pipeline.runner --skip collect,extract,combine,train,export
```

### Configure Deduplication

```yaml
processing:
  deduplication:
    min_output_length: 50      # Minimum response length
    max_repetition: 3          # Max repeated lines
    similarity_threshold: 0.95 # Semantic similarity cutoff
```

### Check Output

```bash
# Compare before/after
echo "Combined: $(wc -l < data/training/combined.json) examples"
echo "Deduped:  $(wc -l < data/training/deduped.json) examples"

# View deduped examples
head -3 data/training/deduped.json | python -m json.tool
```

---

## Stage 5: TRAIN (with Auto-Recovery)

**Purpose**: Fine-tune LLM on training data with automatic GPU hang recovery

**Input**: `/data/training/deduped.json` from Stage 4
**Output**: `/finetune/output/{model_name}/checkpoint-*/`
**Duration**: 30 minutes to several hours (depends on data size and GPU)
**Special**: Automatic GPU hang detection and recovery

### Run Stage 5 Independently

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Train with auto-recovery (recommended)
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
tail -f /tmp/auto_recovery.log

# Or systemd service
cd finetune
sudo ./setup_auto_recovery.sh
sudo systemctl start training-recovery
sudo journalctl -u training-recovery -f
```

### Configure Training

Edit `config/pipeline_config.yaml`:

```yaml
training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  num_gpus: 4

  lora:
    r: 8
    alpha: 16
    dropout: 0.1

  hyperparameters:
    learning_rate: 0.00005
    num_epochs: 4
    micro_batch_size: 1
    gradient_accumulation_steps: 4
    sequence_len: 1536
```

### Monitor Training

```bash
# View recovery logs
tail -f /tmp/training_recovery/recovery_*.log

# Check training progress
python3 -c "
import json
with open('finetune/output/terraform-expert-v2/trainer_state.json') as f:
    state = json.load(f)
    step = state['global_step']
    total = 3495
    pct = 100 * step / total
    print(f'Progress: {step}/{total} ({pct:.1f}%)')
    print(f'Remaining: {total - step} steps')
"

# GPU status
rocm-smi
```

### Auto-Recovery Details

See `TRAINING_AUTO_RECOVERY.md` for complete auto-recovery documentation:
- How recovery works
- Recovery levels
- Troubleshooting
- Configuration options

---

## Stage 6: EXPORT

**Purpose**: Merge LoRA adapter with base model, convert to GGUF, create Ollama model

**Input**: Checkpoint from Stage 5 (`/finetune/output/{model_name}/checkpoint-*/`)
**Output**: Ollama model ready for inference
**Duration**: 5-15 minutes depending on model size

### Run Stage 6 Independently

```bash
cd /home/paulpas/git/ideas/llm_training_web_data

# Export only (must have completed training)
python -m pipeline.runner --stage export

# Or with skipping
python -m pipeline.runner --skip collect,extract,combine,dedupe,train
```

### Configure Export

Edit `config/pipeline_config.yaml`:

```yaml
export:
  merge:
    torch_dtype: float16
    device_map: cpu

  gguf:
    quantization: q4_k_m        # q4_k_m, q5_k_m, q8_0, f16

  ollama:
    temperature: 0.7
    repeat_penalty: 1.1
    num_predict: 512
    system_prompt: "You are a Python programming expert."
```

### Check Output

```bash
# List export outputs
ls -la finetune/output/terraform-expert-v2/

# Check Ollama model
ollama list | grep terraform

# Test model
ollama run terraform-expert "What is Python?"
```

---

## Running Stages in Sequence

### Complete Pipeline (All Stages)

```bash
cd /home/paulpas/git/ideas/llm_training_web_data
python -m pipeline.runner --config config/pipeline_config.yaml
```

### Common Workflows

#### Workflow 1: Data Processing Only

Skip training and export:
```bash
python -m pipeline.runner --skip train,export
```
Output: `/data/training/deduped.json` ready for manual training

#### Workflow 2: Training Only

Skip data collection (use existing data):
```bash
python -m pipeline.runner --skip collect,extract,combine,dedupe
```
Or with auto-recovery:
```bash
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
```

#### Workflow 3: Resume Training

Training auto-recovers from latest checkpoint. To manually resume:
```bash
# Current checkpoint is automatically detected
nohup ./finetune/start_auto_recovery.sh > /tmp/auto_recovery.log 2>&1 &
```

#### Workflow 4: Export Only

After training completes:
```bash
python -m pipeline.runner --stage export
```

#### Workflow 5: Process New Repository

Add new source and skip old data:
```bash
# 1. Edit config/pipeline_config.yaml
#    - Add new git_source URL
#
# 2. Run only new extraction and beyond
python -m pipeline.runner --stage extract,combine,dedupe,train,export
```

---

## Command Reference

### Run All Stages
```bash
python -m pipeline.runner
```

### Run Specific Stages
```bash
python -m pipeline.runner --stage collect,extract
python -m pipeline.runner --stage train,export
python -m pipeline.runner --stage export
```

### Skip Specific Stages
```bash
python -m pipeline.runner --skip train,export    # Skip training & export
python -m pipeline.runner --skip collect         # Skip collection
```

### Use Profile
```bash
python -m pipeline.runner --profile test         # Use 'test' profile
python -m pipeline.runner --profile production   # Use 'production' profile
```

### Override Configuration
```bash
export PIPELINE_TRAINING_NUM_GPUS=2
export PIPELINE_TRAINING_BASE_MODEL="mistral-ai/Mistral-7B"
python -m pipeline.runner
```

---

## File Structure

```
llm_training_web_data/
├── AGENTS.md                          # Complete reference
├── PIPELINE_STAGES.md                 # This file
├── TRAINING_AUTO_RECOVERY.md          # Training with recovery
├── QUICK_REFERENCE.md                 # Common commands
│
├── config/
│   └── pipeline_config.yaml           # Configuration
│
├── data/
│   ├── extracted/
│   │   ├── k8s_extracted.json         # Stage 2 outputs
│   │   └── pytest_extracted.json
│   └── training/
│       ├── combined.json              # Stage 3 output
│       └── deduped.json               # Stage 4 output
│
├── finetune/
│   ├── auto_recover.py                # Auto-recovery engine
│   ├── start_auto_recovery.sh         # Launch recovery
│   ├── setup_auto_recovery.sh         # Setup systemd
│   ├── training-recovery.service      # Systemd config
│   ├── axolotl_config_v2_resume.yaml  # Training config
│   └── output/
│       └── terraform-expert-v2/
│           ├── checkpoint-*/          # Stage 5 outputs
│           └── trainer_state.json
│
├── pipeline/
│   ├── runner.py                      # Main orchestrator
│   ├── config_loader.py               # Config management
│   ├── data_extractor.py              # Stage 2 implementation
│   ├── data_processor.py              # Stages 3 & 4
│   └── model_exporter.py              # Stage 6 implementation
│
├── repos/                             # Stage 1 outputs
│   ├── k8s-python/
│   └── pytest/
│
└── requirements.txt
```

---

## Troubleshooting by Stage

### Stage 1: COLLECT
**Issue**: Repository not found
**Solution**: Verify URL in config, check internet connection

**Issue**: Partial clone
**Solution**: Delete `/repos/{name}` and retry

### Stage 2: EXTRACT
**Issue**: No examples generated
**Solution**: Check source type (python/mixed), verify file patterns

**Issue**: OOM during extraction
**Solution**: Process smaller repos first

### Stage 3: COMBINE
**Issue**: Missing files
**Solution**: Run extract stage first

### Stage 4: DEDUPLICATE
**Issue**: All examples removed
**Solution**: Relax filters (increase min_output_length, max_repetition)

### Stage 5: TRAIN
**Issue**: GPU hang
**Solution**: Auto-recovery handles this. See `TRAINING_AUTO_RECOVERY.md`

**Issue**: OOM
**Solution**: Reduce micro_batch_size or sequence_len in config

### Stage 6: EXPORT
**Issue**: Merge fails
**Solution**: Ensure training completed, check checkpoint exists

**Issue**: GGUF conversion fails
**Solution**: Check PyTorch version, ensure llama-cpp-python installed

---

## Next Steps

1. **Run complete pipeline**: `python -m pipeline.runner`
2. **Monitor training**: `tail -f /tmp/training_recovery/recovery_*.log`
3. **Use trained model**: `ollama run terraform-expert`

For detailed training documentation, see `TRAINING_AUTO_RECOVERY.md`.
