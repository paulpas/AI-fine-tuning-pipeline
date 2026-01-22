# Automated LLM Fine-Tuning Pipeline

This tutorial provides a single script to run the entire fine-tuning pipeline from start to finish.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Configuration](#pipeline-configuration)
3. [Pipeline Script](#pipeline-script)
4. [Usage Examples](#usage-examples)
5. [Monitoring and Logs](#monitoring-and-logs)

---

## Quick Start

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo>
cd llm_training_web_data
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure your training (edit config file)
vim config/pipeline_config.yaml

# Run the full pipeline with the unified runner
uv run python -m pipeline.runner --config config/pipeline_config.yaml
```

---

## Pipeline Configuration

Create `finetune/pipeline_config.yaml`:

```yaml
# =============================================================================
# LLM Fine-Tuning Pipeline Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Project Settings
# -----------------------------------------------------------------------------
project:
  name: "my-fine-tuned-model"
  output_dir: "finetune/output"

# -----------------------------------------------------------------------------
# GPU Settings
# -----------------------------------------------------------------------------
gpu:
  platform: "rocm"  # Options: rocm, cuda
  device_ids: "0,1,2,3"
  # ROCm-specific settings (only for AMD GPUs)
  rocm:
    gfx_version: "9.0.6"  # For MI50. Set to null for modern GPUs

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
data:
  # Raw data input
  raw_data_path: "data/raw/dataset.json"

  # Processed data output
  training_data_path: "data/training/alpaca.json"
  cleaned_data_path: "data/training/alpaca_deduped.json"

  # Cleaning settings
  min_output_length: 50
  max_repetition_count: 3

# -----------------------------------------------------------------------------
# Base Model
# -----------------------------------------------------------------------------
model:
  name: "deepseek-ai/deepseek-coder-6.7b-instruct"
  type: "LlamaForCausalLM"
  trust_remote_code: true

# -----------------------------------------------------------------------------
# LoRA Configuration
# -----------------------------------------------------------------------------
lora:
  rank: 8
  alpha: 16
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
training:
  epochs: 4
  learning_rate: 0.00005
  weight_decay: 0.05
  max_grad_norm: 0.5
  warmup_ratio: 0.1
  lr_scheduler: "cosine"

  # Batch settings
  micro_batch_size: 2
  gradient_accumulation_steps: 8

  # Sequence settings
  sequence_length: 2048
  sample_packing: true

  # Evaluation
  val_set_size: 0.1
  eval_steps: 50
  save_steps: 50

  # Early stopping
  early_stopping_patience: 3

# -----------------------------------------------------------------------------
# Model Conversion
# -----------------------------------------------------------------------------
conversion:
  # GGUF quantization type
  quantization: "q4_k_m"  # Options: q4_k_m, q5_k_m, q8_0, f16

  # Ollama model name
  ollama_model_name: "my-model"

  # Inference parameters
  inference:
    temperature: 0.5
    repeat_penalty: 1.3
    num_predict: 512

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
evaluation:
  # Base model for comparison
  base_model_ollama: "deepseek-coder:6.7b-instruct"

  # Number of test questions
  num_tests: 10

  # Timeout per question (seconds)
  timeout: 60

# -----------------------------------------------------------------------------
# Pipeline Steps (enable/disable)
# -----------------------------------------------------------------------------
steps:
  prepare_data: true
  deduplicate: true
  train: true
  merge_lora: true
  convert_gguf: true
  import_ollama: true
  evaluate: true
```

---

## Pipeline Script

Create `scripts/run_pipeline.sh`:

```bash
#!/bin/bash
# =============================================================================
# LLM Fine-Tuning Pipeline
# Runs all steps: data prep, training, conversion, evaluation
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default config
CONFIG_FILE="finetune/pipeline_config.yaml"
LOG_DIR="finetune/logs"
DRY_RUN=false
SKIP_CONFIRM=false

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}STEP: $1${NC}"
    echo "============================================================"
}

# Parse YAML (simple parser for flat values)
parse_yaml() {
    local yaml_file=$1
    local key=$2
    python3 -c "
import yaml
with open('$yaml_file') as f:
    config = yaml.safe_load(f)
keys = '$key'.split('.')
value = config
for k in keys:
    value = value.get(k, {})
print(value if value else '')
"
}

check_step_enabled() {
    local step=$1
    local enabled=$(parse_yaml "$CONFIG_FILE" "steps.$step")
    [[ "$enabled" == "True" || "$enabled" == "true" ]]
}

# =============================================================================
# Parse Arguments
# =============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE    Path to pipeline config (default: $CONFIG_FILE)"
    echo "  -d, --dry-run        Show what would be done without executing"
    echo "  -y, --yes            Skip confirmation prompts"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run with default config"
    echo "  $0 -c my_config.yaml            # Run with custom config"
    echo "  $0 --dry-run                    # Show pipeline steps"
    echo "  $0 -y                           # Run without confirmation"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Environment
# =============================================================================

log_step "Validating Environment"

# Check config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    log_warning "No virtual environment detected"
    if [[ -f ".venv/bin/activate" ]]; then
        log_info "Activating .venv..."
        source .venv/bin/activate
    else
        log_error "Please activate a virtual environment first"
        exit 1
    fi
fi

# Load configuration
PROJECT_NAME=$(parse_yaml "$CONFIG_FILE" "project.name")
OUTPUT_DIR=$(parse_yaml "$CONFIG_FILE" "project.output_dir")
GPU_PLATFORM=$(parse_yaml "$CONFIG_FILE" "gpu.platform")
DEVICE_IDS=$(parse_yaml "$CONFIG_FILE" "gpu.device_ids")

log_info "Project: $PROJECT_NAME"
log_info "Output directory: $OUTPUT_DIR"
log_info "GPU platform: $GPU_PLATFORM"
log_info "Device IDs: $DEVICE_IDS"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "data/training"

# =============================================================================
# Setup GPU Environment
# =============================================================================

log_step "Setting Up GPU Environment"

if [[ "$GPU_PLATFORM" == "rocm" ]]; then
    GFX_VERSION=$(parse_yaml "$CONFIG_FILE" "gpu.rocm.gfx_version")
    if [[ -n "$GFX_VERSION" && "$GFX_VERSION" != "None" ]]; then
        export HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION"
        log_info "Set HSA_OVERRIDE_GFX_VERSION=$GFX_VERSION"
    fi
    export ROCR_VISIBLE_DEVICES="$DEVICE_IDS"
    export HIP_VISIBLE_DEVICES="$DEVICE_IDS"
    log_info "ROCm environment configured"
else
    export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
    log_info "CUDA environment configured"
fi

# Verify GPU access
log_info "Verifying GPU access..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

if [[ $? -ne 0 ]]; then
    log_error "GPU verification failed"
    exit 1
fi
log_success "GPU environment verified"

# =============================================================================
# Confirmation
# =============================================================================

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "DRY RUN - Pipeline Steps:"
    echo "============================================================"
    check_step_enabled "prepare_data" && echo "  1. [ENABLED] Prepare data"
    check_step_enabled "deduplicate" && echo "  2. [ENABLED] Deduplicate data"
    check_step_enabled "train" && echo "  3. [ENABLED] Train model"
    check_step_enabled "merge_lora" && echo "  4. [ENABLED] Merge LoRA"
    check_step_enabled "convert_gguf" && echo "  5. [ENABLED] Convert to GGUF"
    check_step_enabled "import_ollama" && echo "  6. [ENABLED] Import to Ollama"
    check_step_enabled "evaluate" && echo "  7. [ENABLED] Evaluate model"
    echo ""
    exit 0
fi

if [[ "$SKIP_CONFIRM" != "true" ]]; then
    echo ""
    echo "Ready to start pipeline for: $PROJECT_NAME"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
fi

# =============================================================================
# Step 1: Data Preparation
# =============================================================================

if check_step_enabled "prepare_data"; then
    log_step "Step 1: Data Preparation"

    RAW_DATA=$(parse_yaml "$CONFIG_FILE" "data.raw_data_path")
    TRAINING_DATA=$(parse_yaml "$CONFIG_FILE" "data.training_data_path")

    if [[ -f "$RAW_DATA" ]]; then
        log_info "Converting raw data to training format..."
        python3 scripts/prepare_data.py \
            --input "$RAW_DATA" \
            --output "$TRAINING_DATA" \
            2>&1 | tee "$LOG_DIR/01_prepare_data.log"
        log_success "Data preparation complete"
    else
        log_warning "Raw data not found at $RAW_DATA, skipping preparation"
    fi
fi

# =============================================================================
# Step 2: Data Deduplication
# =============================================================================

if check_step_enabled "deduplicate"; then
    log_step "Step 2: Data Deduplication"

    TRAINING_DATA=$(parse_yaml "$CONFIG_FILE" "data.training_data_path")
    CLEANED_DATA=$(parse_yaml "$CONFIG_FILE" "data.cleaned_data_path")
    MIN_LENGTH=$(parse_yaml "$CONFIG_FILE" "data.min_output_length")
    MAX_REP=$(parse_yaml "$CONFIG_FILE" "data.max_repetition_count")

    if [[ -f "$TRAINING_DATA" ]]; then
        log_info "Deduplicating dataset..."
        python3 scripts/deduplicate_dataset.py \
            --input "$TRAINING_DATA" \
            --output "$CLEANED_DATA" \
            --min-output-length "${MIN_LENGTH:-50}" \
            --max-repetition "${MAX_REP:-3}" \
            2>&1 | tee "$LOG_DIR/02_deduplicate.log"
        log_success "Deduplication complete"
    else
        log_error "Training data not found at $TRAINING_DATA"
        exit 1
    fi
fi

# =============================================================================
# Step 3: Training
# =============================================================================

if check_step_enabled "train"; then
    log_step "Step 3: Training"

    # Generate Axolotl config from pipeline config
    log_info "Generating training configuration..."
    python3 scripts/generate_axolotl_config.py \
        --pipeline-config "$CONFIG_FILE" \
        --output "finetune/axolotl_config_generated.yaml"

    # Clear cache
    rm -rf finetune/prepared_data

    # Start training
    log_info "Starting training..."
    TRAINING_LOG="$LOG_DIR/03_training.log"

    accelerate launch -m axolotl.cli.train \
        finetune/axolotl_config_generated.yaml \
        2>&1 | tee "$TRAINING_LOG"

    # Verify training success
    if grep -q "Saving model" "$TRAINING_LOG" || grep -q "Training complete" "$TRAINING_LOG"; then
        log_success "Training completed successfully"
    else
        log_error "Training may have failed. Check $TRAINING_LOG"
        exit 1
    fi
fi

# =============================================================================
# Step 4: Merge LoRA
# =============================================================================

if check_step_enabled "merge_lora"; then
    log_step "Step 4: Merging LoRA Adapter"

    BASE_MODEL=$(parse_yaml "$CONFIG_FILE" "model.name")
    LORA_PATH="$OUTPUT_DIR/$PROJECT_NAME"
    MERGED_PATH="$OUTPUT_DIR/$PROJECT_NAME-merged"

    log_info "Merging LoRA adapter with base model..."
    python3 scripts/merge_lora.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$LORA_PATH" \
        --output "$MERGED_PATH" \
        2>&1 | tee "$LOG_DIR/04_merge_lora.log"

    log_success "LoRA merge complete"
fi

# =============================================================================
# Step 5: Convert to GGUF
# =============================================================================

if check_step_enabled "convert_gguf"; then
    log_step "Step 5: Converting to GGUF"

    MERGED_PATH="$OUTPUT_DIR/$PROJECT_NAME-merged"
    GGUF_PATH="$OUTPUT_DIR/$PROJECT_NAME-gguf"
    QUANTIZATION=$(parse_yaml "$CONFIG_FILE" "conversion.quantization")

    log_info "Converting to GGUF format (quantization: $QUANTIZATION)..."
    python3 scripts/convert_to_gguf.py \
        --model-path "$MERGED_PATH" \
        --output-path "$GGUF_PATH" \
        --quantization "${QUANTIZATION:-q4_k_m}" \
        2>&1 | tee "$LOG_DIR/05_convert_gguf.log"

    log_success "GGUF conversion complete"
fi

# =============================================================================
# Step 6: Import to Ollama
# =============================================================================

if check_step_enabled "import_ollama"; then
    log_step "Step 6: Importing to Ollama"

    GGUF_PATH="$OUTPUT_DIR/$PROJECT_NAME-gguf"
    OLLAMA_NAME=$(parse_yaml "$CONFIG_FILE" "conversion.ollama_model_name")
    TEMP=$(parse_yaml "$CONFIG_FILE" "conversion.inference.temperature")
    REP_PENALTY=$(parse_yaml "$CONFIG_FILE" "conversion.inference.repeat_penalty")
    NUM_PREDICT=$(parse_yaml "$CONFIG_FILE" "conversion.inference.num_predict")

    log_info "Creating Ollama model: $OLLAMA_NAME..."
    python3 scripts/create_ollama_model.py \
        --gguf-path "$GGUF_PATH/model-*.gguf" \
        --output-dir "$GGUF_PATH" \
        --model-name "$OLLAMA_NAME" \
        --temperature "${TEMP:-0.5}" \
        --repeat-penalty "${REP_PENALTY:-1.3}" \
        --num-predict "${NUM_PREDICT:-512}" \
        2>&1 | tee "$LOG_DIR/06_import_ollama.log"

    log_success "Ollama import complete"
    log_info "Model available as: ollama run $OLLAMA_NAME"
fi

# =============================================================================
# Step 7: Evaluation
# =============================================================================

if check_step_enabled "evaluate"; then
    log_step "Step 7: Evaluation"

    OLLAMA_NAME=$(parse_yaml "$CONFIG_FILE" "conversion.ollama_model_name")
    BASE_MODEL=$(parse_yaml "$CONFIG_FILE" "evaluation.base_model_ollama")
    NUM_TESTS=$(parse_yaml "$CONFIG_FILE" "evaluation.num_tests")
    TIMEOUT=$(parse_yaml "$CONFIG_FILE" "evaluation.timeout")

    log_info "Evaluating model performance..."
    python3 scripts/evaluate_model.py \
        --base-model "$BASE_MODEL" \
        --finetuned-model "$OLLAMA_NAME" \
        --num-tests "${NUM_TESTS:-10}" \
        --timeout "${TIMEOUT:-60}" \
        --output "$LOG_DIR/07_evaluation.json" \
        2>&1 | tee "$LOG_DIR/07_evaluation.log"

    log_success "Evaluation complete"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo -e "${GREEN}PIPELINE COMPLETE${NC}"
echo "============================================================"
echo ""
echo "Project: $PROJECT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
[[ -d "$OUTPUT_DIR/$PROJECT_NAME" ]] && echo "  - LoRA adapter: $OUTPUT_DIR/$PROJECT_NAME"
[[ -d "$OUTPUT_DIR/$PROJECT_NAME-merged" ]] && echo "  - Merged model: $OUTPUT_DIR/$PROJECT_NAME-merged"
[[ -d "$OUTPUT_DIR/$PROJECT_NAME-gguf" ]] && echo "  - GGUF model: $OUTPUT_DIR/$PROJECT_NAME-gguf"
echo ""
echo "Logs available in: $LOG_DIR"
echo ""

OLLAMA_NAME=$(parse_yaml "$CONFIG_FILE" "conversion.ollama_model_name")
echo "To use your model:"
echo "  ollama run $OLLAMA_NAME"
echo ""
```

Make the script executable:

```bash
chmod +x scripts/run_pipeline.sh
```

---

## Supporting Scripts

### Generate Axolotl Config

Create `scripts/generate_axolotl_config.py`:

```python
#!/usr/bin/env python3
"""
Generate Axolotl config from pipeline config.
"""

import yaml
import argparse
from pathlib import Path

def generate_config(pipeline_config: str, output_path: str):
    """Generate Axolotl config from pipeline config."""
    with open(pipeline_config) as f:
        config = yaml.safe_load(f)

    project = config['project']
    model = config['model']
    lora = config['lora']
    training = config['training']
    data = config['data']

    axolotl_config = {
        # Base model
        'base_model': model['name'],
        'model_type': model['type'],
        'tokenizer_type': 'AutoTokenizer',
        'trust_remote_code': model.get('trust_remote_code', True),

        # Dataset
        'datasets': [{
            'path': data['cleaned_data_path'],
            'type': 'alpaca'
        }],

        # Output
        'output_dir': f"{config['project']['output_dir']}/{project['name']}",

        # LoRA
        'adapter': 'lora',
        'lora_r': lora['rank'],
        'lora_alpha': lora['alpha'],
        'lora_dropout': lora['dropout'],
        'lora_target_linear': True,
        'lora_target_modules': lora['target_modules'],

        # Training
        'gradient_accumulation_steps': training['gradient_accumulation_steps'],
        'micro_batch_size': training['micro_batch_size'],
        'num_epochs': training['epochs'],
        'learning_rate': training['learning_rate'],
        'weight_decay': training['weight_decay'],
        'max_grad_norm': training['max_grad_norm'],
        'lr_scheduler': training['lr_scheduler'],
        'warmup_ratio': training['warmup_ratio'],
        'optimizer': 'adamw_torch',

        # Data
        'sequence_len': training['sequence_length'],
        'sample_packing': training['sample_packing'],
        'pad_to_sequence_len': True,
        'val_set_size': training['val_set_size'],

        # Evaluation
        'eval_steps': training['eval_steps'],
        'save_steps': training['save_steps'],
        'save_strategy': 'steps',
        'eval_strategy': 'steps',
        'save_total_limit': 5,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'early_stopping_patience': training['early_stopping_patience'],

        # Precision
        'bf16': True,
        'tf32': False,
        'flash_attention': False,

        # Logging
        'logging_steps': 10,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(axolotl_config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated Axolotl config: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    generate_config(args.pipeline_config, args.output)
```

---

## Usage Examples

### Run Full Pipeline

```bash
# Default configuration
./scripts/run_pipeline.sh

# Custom configuration
./scripts/run_pipeline.sh -c finetune/my_config.yaml

# Skip confirmation
./scripts/run_pipeline.sh -y

# Dry run (show steps without executing)
./scripts/run_pipeline.sh --dry-run
```

### Run Specific Steps Only

Edit the `steps` section in your config:

```yaml
steps:
  prepare_data: false    # Skip - data already prepared
  deduplicate: false     # Skip - already cleaned
  train: true            # Run training
  merge_lora: true       # Run merge
  convert_gguf: true     # Run conversion
  import_ollama: true    # Import to Ollama
  evaluate: true         # Run evaluation
```

### Training Only (No Conversion)

```yaml
steps:
  prepare_data: true
  deduplicate: true
  train: true
  merge_lora: false
  convert_gguf: false
  import_ollama: false
  evaluate: false
```

---

## Monitoring and Logs

### Real-time Monitoring

```bash
# Watch training progress
tail -f finetune/logs/03_training.log

# Monitor GPU usage (AMD)
watch -n 2 rocm-smi --showuse

# Monitor GPU usage (NVIDIA)
watch -n 2 nvidia-smi
```

### Log Files

All logs are saved in `finetune/logs/`:

| File | Description |
|------|-------------|
| `01_prepare_data.log` | Data preparation output |
| `02_deduplicate.log` | Deduplication statistics |
| `03_training.log` | Full training log |
| `04_merge_lora.log` | LoRA merge output |
| `05_convert_gguf.log` | GGUF conversion |
| `06_import_ollama.log` | Ollama import |
| `07_evaluation.log` | Evaluation results |
| `07_evaluation.json` | Detailed evaluation metrics |

### Check Pipeline Status

```bash
# View all logs
ls -la finetune/logs/

# Check for errors
grep -i "error" finetune/logs/*.log

# Check training metrics
grep -E "loss|eval_loss" finetune/logs/03_training.log | tail -20
```

---

## Troubleshooting

### Pipeline Fails at Training

1. Check GPU memory:
   ```bash
   rocm-smi --showmeminfo vram  # AMD
   nvidia-smi                   # NVIDIA
   ```

2. Reduce batch size in config:
   ```yaml
   training:
     micro_batch_size: 1
     gradient_accumulation_steps: 16
   ```

### Pipeline Fails at Conversion

1. Ensure llama.cpp is installed:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
   cd ~/llama.cpp && make -j
   ```

2. Check disk space for merged model (~15GB)

### Model Has Repetition Issues

1. Increase repetition penalty:
   ```yaml
   conversion:
     inference:
       repeat_penalty: 1.5
   ```

2. Lower learning rate and retrain:
   ```yaml
   training:
     learning_rate: 0.00002
   ```

---

## Next Steps

- Review [02-STEP-BY-STEP-PIPELINE.md](./02-STEP-BY-STEP-PIPELINE.md) for detailed explanations
- Check [04-TROUBLESHOOTING.md](./04-TROUBLESHOOTING.md) for common issues
- Customize the pipeline for your specific use case
