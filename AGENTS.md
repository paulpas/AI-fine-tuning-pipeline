# LLM Fine-Tuning Pipeline - Complete Reference

**A modularized, production-ready LLM training pipeline with automatic GPU utilization, flexible data sources, and turn-key deployment.**

## Quick Start (60 seconds)

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure pipeline (optional)
# Edit config/pipeline_config.yaml:
#   - Set base_model (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
#   - Add git_sources (GitHub repos to train on)
#   - Adjust training parameters (num_gpus, batch_size, epochs)

# 3. Run complete pipeline (data → training → export)
python -m pipeline.runner --config config/pipeline_config.yaml

# 4. Use the trained model
ollama run python-expert  # Or your configured model name
```

## Quick Links

- **📊 Running Everything**: `python -m pipeline.runner` (see QUICK_REFERENCE.md)
- **🚀 Training with Auto-Recovery**: `./finetune/start_auto_recovery.sh` (see TRAINING_AUTO_RECOVERY.md)
- **📋 Each Stage Independently**: See PIPELINE_STAGES.md
- **⚡ Common Commands**: See QUICK_REFERENCE.md

---

## Full Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│              LLM FINE-TUNING PIPELINE (6 STAGES)                  │
└──────────────────────────────────────────────────────────────────┘

STAGE 1: COLLECT
  ├─ Clone GitHub repositories
  └─ Output: /repos/{repo_name}/

STAGE 2: EXTRACT
  ├─ Parse Python code (AST)
  ├─ Extract documentation (RST/Markdown)
  └─ Output: /data/extracted/{source_name}.json

STAGE 3: COMBINE
  ├─ Merge all datasets
  └─ Output: /data/training/combined.json

STAGE 4: DEDUPLICATE
  ├─ Remove duplicates & low-quality examples
  └─ Output: /data/training/deduped.json

STAGE 5: TRAIN (with Auto-Recovery)
  ├─ Fine-tune model with Axolotl
  ├─ Use all available GPUs automatically (DDP)
  ├─ Auto-detect GPU hangs, crashes, stalls
  ├─ Auto-recover with progressively conservative settings (5 retries)
  ├─ Save checkpoints every N steps
  └─ Output: /finetune/output/{model_name}/checkpoint-*/

  AUTO-RECOVERY DETAILS:
  ├─ Detects: GPU hangs (kernel), process crashes, progress stalls
  ├─ Recovers: Kill zombies → Reset GPU → Use conservative config → Resume
  ├─ Levels: 4 recovery levels with progressively safer settings
  ├─ Retries: Up to 5 automatic recovery attempts
  └─ Logs: /tmp/training_recovery/recovery_*.log

STAGE 6: EXPORT
  ├─ Merge LoRA adapter with base model
  ├─ Convert to GGUF (quantized)
  ├─ Import to Ollama
  └─ Output: Ollama model ready for inference
```

---

## Command Reference

### Run All Stages
```bash
python -m pipeline.runner --config config/pipeline_config.yaml
```

### Run Specific Stages
```bash
# Only training and export (skip data collection)
python -m pipeline.runner --stage train,export

# Only data processing (no training)
python -m pipeline.runner --stage collect,extract,combine,dedupe
```

### Skip Stages
```bash
# Skip data collection (use existing data)
python -m pipeline.runner --skip collect,extract
```

### Environment Variable Overrides
```bash
# Override number of GPUs
export PIPELINE_TRAINING_NUM_GPUS=2

# Override base model
export PIPELINE_TRAINING_BASE_MODEL="mistral-ai/Mistral-7B"

python -m pipeline.runner
```

---

## Configuration Guide

### Minimal Config (config/pipeline_config.yaml)

```yaml
pipeline:
  name: my-model
  description: My custom model

git_sources:
  - name: my-repo
    url: https://github.com/user/repo
    enabled: true

training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  num_gpus: 4           # Auto-uses all available GPUs
  num_epochs: 3
```

### Full Config Example

```yaml
pipeline:
  name: python-expert
  version: v1
  description: Python programming expert model

# Path configuration
paths:
  root: "."
  repos: "repos"
  data:
    training: "data/training"
    combined: "data/training/combined.json"
    deduped: "data/training/deduped.json"
  output:
    checkpoints: "finetune/output"

# Data sources
git_sources:
  - name: k8s-python
    url: https://github.com/kubernetes-client/python
    subdirs: [examples, docs]
    type: python
    enabled: true

  - name: pytest
    url: https://github.com/pytest-dev/pytest
    subdirs: [doc/en]
    type: mixed          # Python + RST
    enabled: true

# Training configuration
training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  chat_template: auto   # auto-detect or: deepseek, gemma, chatml, llama2, phi
  num_gpus: 4          # Uses all available GPUs via DDP

  # LoRA fine-tuning
  lora:
    r: 16               # LoRA rank (larger = more params, slower)
    alpha: 32           # Alpha scaling
    dropout: 0.1
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

  # Hyperparameters
  hyperparameters:
    learning_rate: 0.00003
    num_epochs: 3
    micro_batch_size: 4  # Per-GPU batch (reduce if OOM)
    gradient_accumulation_steps: 2
    sequence_len: 2048   # Max token length per sample
    warmup_ratio: 0.1
    weight_decay: 0.1
    max_grad_norm: 0.5

  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    metric: eval_loss

# Data processing
processing:
  min_code_length: 50
  max_code_length: 10000
  output_format: alpaca

  deduplication:
    min_output_length: 50
    max_repetition: 3

# Export settings
export:
  merge:
    torch_dtype: float16
    device_map: cpu
  gguf:
    quantization: q4_k_m   # q4_k_m, q5_k_m, q8_0, f16
  ollama:
    temperature: 0.7
    repeat_penalty: 1.1
    num_predict: 512
    system_prompt: "You are a Python programming expert."

# Profiles for different configurations
profiles:
  test:
    training:
      hyperparameters:
        num_epochs: 1
        micro_batch_size: 2

  production:
    training:
      hyperparameters:
        num_epochs: 5
      lora:
        r: 32
        alpha: 64
```

---

## Hardware Optimization

The pipeline **automatically detects and uses all available GPUs** via distributed data parallel (DDP).

### Memory Tuning Guide

If you encounter GPU memory issues or hangs during multi-GPU training:

**1. Reduce sequence length** (tokens per sample):
```yaml
training:
  hyperparameters:
    sequence_len: 1024  # From 2048
```

**2. Reduce batch size per GPU**:
```yaml
training:
  hyperparameters:
    micro_batch_size: 1  # From 4
```

**3. Increase gradient accumulation** (keeps effective batch size):
```yaml
training:
  hyperparameters:
    gradient_accumulation_steps: 16  # From 2
```

**Combined example** (4-GPU ultra-stable):
```yaml
training:
  hyperparameters:
    sequence_len: 512
    micro_batch_size: 1
    gradient_accumulation_steps: 16
```

This maintains effective batch size while reducing per-GPU memory from 8GB to under 2GB.

### GPU Status Monitoring

**AMD ROCm:**
```bash
watch -n 1 rocm-smi
```

**NVIDIA CUDA:**
```bash
watch -n 1 nvidia-smi
```

**Check individual GPU:**
```bash
rocm-smi --query=count,id,name,temp,power  # AMD
nvidia-smi -l 1                             # NVIDIA
```

---

## Modular Code Architecture

The pipeline uses **independent, reusable stages** that can be:
- Run individually
- Modified without affecting others
- Extended with custom implementations

### Core Modules

**`pipeline/runner.py`** - Main orchestrator
```python
# Runs stages in sequence with error handling
STAGES = {
    "collect": stage_collect,      # Clone repos
    "extract": stage_extract,      # Parse code
    "combine": stage_combine,      # Merge datasets
    "dedupe": stage_dedupe,        # Remove duplicates
    "train": stage_train,          # Fine-tune model
    "export": stage_export,        # Export to Ollama
}
```

**`pipeline/config_loader.py`** - Configuration management
```python
# Loads YAML, validates, applies profiles, env overrides
config = load_config("config/pipeline_config.yaml")
config.apply_profile("production")
```

**`pipeline/data_extractor.py`** - Extract training data
```python
# Extracts Python code, docstrings, RST docs
extract_from_git_source(source_config) → training_examples
```

**`pipeline/data_processor.py`** - Process datasets
```python
# Deduplication, quality filtering, combining
deduplicate_dataset(input_json, output_json)
```

**`pipeline/model_exporter.py`** - Export trained model
```python
# Merge LoRA, convert GGUF, create Ollama model
export_model(checkpoint_path, base_model)
```

### Adding Custom Stages

**1. Create new stage function in `pipeline/runner.py`:**
```python
def stage_custom(config: PipelineConfig) -> StageResult:
    """Custom pipeline stage."""
    start = datetime.now()
    log.info("STAGE X: CUSTOM OPERATION")

    # Your implementation
    success = do_something_useful(config)

    duration = (datetime.now() - start).total_seconds()
    return StageResult(
        stage="custom",
        success=success,
        duration=duration,
        message="Custom operation completed"
    )
```

**2. Register in STAGES and STAGE_ORDER:**
```python
STAGES["custom"] = stage_custom
STAGE_ORDER = ["collect", "extract", "custom", "combine", ...]
```

**3. Run it:**
```bash
python -m pipeline.runner --stage custom
```

### Extending Configurations

Add new config sections as needed:

```yaml
custom_processing:
  quality_threshold: 0.8
  min_tokens: 100

custom_models:
  encoder: sentence-transformers/all-MiniLM-L6-v2
```

Load in your code:
```python
config = load_config("config/pipeline_config.yaml")
quality = config.custom_processing.quality_threshold
```

---

## Supported Models

### Model Families

| Family | Example | Chat Template |
|--------|---------|--------------|
| **DeepSeek** | DeepSeek-R1-Distill-Qwen-1.5B | `deepseek` |
| **CodeGemma** | google/codegemma-2b | `gemma` |
| **Qwen** | Qwen2-7B | `chatml` |
| **LLaMA/Mistral** | mistral-ai/Mistral-7B | `llama2` |
| **Phi** | microsoft/phi-2 | `phi` |

### Auto-Detection

The pipeline auto-detects model family from HuggingFace model ID:
```yaml
training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  chat_template: auto  # Automatically detects "deepseek"
```

### Manual Override

```yaml
training:
  base_model: custom-model/custom-7b
  chat_template: chatml  # Force specific template
```

---

## Monitoring Training

### Real-time Progress

```bash
# Watch log file
tail -f finetune/output/python-expert-v5/training.log

# Or check specific metrics
grep "step\|loss" training.log | tail -20
```

### Check Checkpoints

```bash
# List all checkpoints
ls finetune/output/python-expert-v5/checkpoint-*/

# Get step count from latest
python3 -c "import json; ckpt = json.load(open('finetune/output/python-expert-v5/checkpoint-*/trainer_state.json')); print(ckpt['global_step'])"
```

### Resume from Checkpoint

The pipeline automatically detects and resumes from the latest checkpoint:
```bash
# Automatically resumes from checkpoint-2500 if it exists
python -m pipeline.runner --stage train
```

---

## Troubleshooting

### GPU Out of Memory

**Error:** `CUDA out of memory` or `GPU Hang`

**Solution:** Reduce per-GPU memory usage
```yaml
training:
  hyperparameters:
    sequence_len: 512         # Reduce from 2048
    micro_batch_size: 1       # Reduce from 4
    gradient_accumulation_steps: 16  # Increase
```

### Training Loss Not Decreasing

**Solutions:**
1. Increase learning rate: `0.0001` (was `0.00003`)
2. Increase epochs: `num_epochs: 5` (was `3`)
3. Check data quality: `head -100 data/training/deduped.json`
4. Increase LoRA rank: `lora.r: 32` (was `16`)

### Multi-GPU Training Issues

**All GPUs not detected:**
```bash
rocm-smi           # AMD: Check GPU count
nvidia-smi         # NVIDIA: Check GPU count
```

**DDP synchronization errors:**
1. Reduce batch size further
2. Reduce sequence length
3. Check GPU drivers are up to date

### Model Quality After Training

**Model generates gibberish:**
1. Check training completed (loss plateaued)
2. Verify GGUF conversion succeeded
3. Adjust Ollama parameters:
   ```yaml
   export:
     ollama:
       temperature: 0.5  # Lower = more focused
       repeat_penalty: 1.2
   ```

### GGUF Conversion Fails

**Error:** `Failed to convert to GGUF`

**Solution:**
1. Ensure PyTorch version matches: `pip list | grep torch`
2. Check llama-cpp-python: `pip install llama-cpp-python`
3. Use float16 instead of float32: `export.merge.torch_dtype: float16`

---

## File Structure

```
llm_training_web_data/
├── AGENTS.md                          # ← You are here
├── config/
│   └── pipeline_config.yaml           # Unified configuration
├── data/
│   └── training/
│       ├── *_extracted.json           # Per-source data
│       ├── combined.json              # Merged data
│       └── deduped.json               # Final training data
├── finetune/
│   ├── axolotl_config_*.yaml          # Generated Axolotl configs
│   └── output/
│       └── {model_name}/
│           ├── checkpoint-*/          # Training checkpoints
│           ├── merged/                # Merged LoRA + base
│           ├── gguf/                  # Quantized GGUF files
│           ├── ollama/                # Ollama model
│           └── *.log                  # Training logs
├── pipeline/                          # Core pipeline module
│   ├── __init__.py
│   ├── runner.py                      # Main orchestrator
│   ├── config_loader.py               # Config management
│   ├── data_extractor.py              # Extract from code
│   ├── data_processor.py              # Dedup & combine
│   └── model_exporter.py              # Merge & export
├── repos/                             # Cloned repositories
├── requirements.txt
└── .venv/                             # Python virtual environment
```

---

## Coding Practices

### Key Design Principles

1. **Modularity**: Each stage is independent
2. **Hardware-First**: Automatically use all available GPUs
3. **Configuration-Driven**: Single YAML file controls everything
4. **Profiles**: Different configs for test/production
5. **Fault-Tolerant**: Resume from checkpoints automatically
6. **Extensible**: Easy to add custom stages

### Error Handling

All stages return `StageResult` for consistent error reporting:
```python
@dataclass
class StageResult:
    stage: str
    success: bool
    duration: float
    message: str = ""
    details: dict = field(default_factory=dict)
```

### Logging

Use structured logging for debugging:
```python
import logging
log = logging.getLogger(__name__)

log.info(f"Stage {stage.name} completed in {duration:.2f}s")
log.error(f"Stage failed: {error}")
log.debug(f"Detailed info: {details}")
```

---

## Common Workflows

### Workflow 1: Train on New Repository

```bash
# 1. Edit config/pipeline_config.yaml
#    - Add git_source with new repo URL
#    - Adjust training parameters

# 2. Run pipeline (skips old data collection)
python -m pipeline.runner --skip extract,combine,dedupe

# 3. Monitor training
tail -f finetune/output/*/training.log
```

### Workflow 2: Quick Testing

```bash
# Use test profile for faster iteration
python -m pipeline.runner --profile test

# Uses: smaller dataset, fewer epochs, faster
```

### Workflow 3: Production Deployment

```bash
# Use production profile for best quality
python -m pipeline.runner --profile production

# Runs: full dataset, 5 epochs, larger LoRA rank
```

### Workflow 4: Tune Hyperparameters

```bash
# Try different learning rates
for lr in 0.00001 0.00003 0.0001; do
  export PIPELINE_TRAINING_LR=$lr
  python -m pipeline.runner --stage train
done

# Compare results in finetune/output/
```

---

## Performance Benchmarks

### Typical Training Times

| Model | Params | GPUs | Batch | Time |
|-------|--------|------|-------|------|
| DeepSeek-1.5B | 1.5B | 1 | 4 | 8-12h |
| DeepSeek-1.5B | 1.5B | 4 | 4 | 2-3h |
| CodeGemma-2B | 2B | 1 | 4 | 6-10h |
| Mistral-7B | 7B | 4 | 2 | 12-24h |

### Memory Requirements

| Model | GPUs | Batch | VRAM/GPU |
|-------|------|-------|----------|
| 1.5B | 1 | 4 | 8GB |
| 1.5B | 4 | 4 | 4GB each |
| 1.5B | 4 | 1 | 2GB each |
| 7B | 1 | 2 | 24GB |
| 7B | 4 | 2 | 12GB each |

---

## Environment Setup

### AMD ROCm GPUs

```bash
# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Set GPU visibility
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Adjust for your GPU
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
```

### NVIDIA CUDA GPUs

```bash
# Install PyTorch for CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# NVIDIA GPUs auto-detected
```

### Full Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU detection
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## Support & Resources

### Debug Commands

```bash
# Check GPU detection
rocm-smi | head -20             # AMD
nvidia-smi                       # NVIDIA

# Verify dependencies
python -c "import axolotl; print(axolotl.__version__)"
python -c "import transformers; print(transformers.__version__)"

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/pipeline_config.yaml'))"

# View training config
cat finetune/output/python-expert-v5/training_args.json
```

### Getting Help

1. Check training log: `tail -100 finetune/output/*/training.log`
2. Review error message in `StageResult.message`
3. Check GPU status: `rocm-smi` or `nvidia-smi`
4. Verify config syntax: `python -m yaml config/pipeline_config.yaml`
5. Test data: `head -5 data/training/deduped.json | python -m json.tool`

---

## License

This pipeline is provided for research and educational purposes. Ensure compliance with:
- Base model licenses (HuggingFace models)
- Training data source licenses (GitHub repositories)
- Framework licenses (Axolotl, PyTorch, Transformers)
