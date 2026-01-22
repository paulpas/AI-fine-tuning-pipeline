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
    if isinstance(value, dict):
        value = value.get(k, {})
    else:
        value = {}
print(value if value and value != {} else '')
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
    if [[ -n "$GFX_VERSION" && "$GFX_VERSION" != "None" && "$GFX_VERSION" != "null" ]]; then
        export HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION"
        log_info "Set HSA_OVERRIDE_GFX_VERSION=$GFX_VERSION"
    fi
    export ROCR_VISIBLE_DEVICES="$DEVICE_IDS"
    export HIP_VISIBLE_DEVICES="$DEVICE_IDS"
    export GPU_DEVICE_ORDINAL="$DEVICE_IDS"
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
    check_step_enabled "prepare_data" && echo "  1. [ENABLED] Prepare data" || echo "  1. [DISABLED] Prepare data"
    check_step_enabled "deduplicate" && echo "  2. [ENABLED] Deduplicate data" || echo "  2. [DISABLED] Deduplicate data"
    check_step_enabled "train" && echo "  3. [ENABLED] Train model" || echo "  3. [DISABLED] Train model"
    check_step_enabled "merge_lora" && echo "  4. [ENABLED] Merge LoRA" || echo "  4. [DISABLED] Merge LoRA"
    check_step_enabled "convert_gguf" && echo "  5. [ENABLED] Convert to GGUF" || echo "  5. [DISABLED] Convert to GGUF"
    check_step_enabled "import_ollama" && echo "  6. [ENABLED] Import to Ollama" || echo "  6. [DISABLED] Import to Ollama"
    check_step_enabled "evaluate" && echo "  7. [ENABLED] Evaluate model" || echo "  7. [DISABLED] Evaluate model"
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

# Record start time
START_TIME=$(date +%s)

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
else
    log_info "Step 1: Data Preparation - SKIPPED"
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
else
    log_info "Step 2: Data Deduplication - SKIPPED"
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
    if grep -qE "(Saving model|Training complete|save_pretrained)" "$TRAINING_LOG"; then
        log_success "Training completed successfully"
    else
        log_error "Training may have failed. Check $TRAINING_LOG"
        exit 1
    fi
else
    log_info "Step 3: Training - SKIPPED"
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
else
    log_info "Step 4: Merge LoRA - SKIPPED"
fi

# =============================================================================
# Step 5: Convert to GGUF
# =============================================================================

if check_step_enabled "convert_gguf"; then
    log_step "Step 5: Converting to GGUF"

    MERGED_PATH="$OUTPUT_DIR/$PROJECT_NAME-merged"
    GGUF_PATH="$OUTPUT_DIR/$PROJECT_NAME-gguf"
    QUANTIZATION=$(parse_yaml "$CONFIG_FILE" "conversion.quantization")

    log_info "Converting to GGUF format (quantization: ${QUANTIZATION:-q4_k_m})..."
    python3 scripts/convert_to_gguf.py \
        --model-path "$MERGED_PATH" \
        --output-path "$GGUF_PATH" \
        --quantization "${QUANTIZATION:-q4_k_m}" \
        2>&1 | tee "$LOG_DIR/05_convert_gguf.log"

    log_success "GGUF conversion complete"
else
    log_info "Step 5: Convert to GGUF - SKIPPED"
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

    # Find GGUF file
    GGUF_FILE=$(ls "$GGUF_PATH"/model-*.gguf 2>/dev/null | head -1)

    if [[ -z "$GGUF_FILE" ]]; then
        log_error "No GGUF file found in $GGUF_PATH"
        exit 1
    fi

    log_info "Creating Ollama model: ${OLLAMA_NAME:-$PROJECT_NAME}..."
    python3 scripts/create_ollama_model.py \
        --gguf-path "$GGUF_FILE" \
        --output-dir "$GGUF_PATH" \
        --model-name "${OLLAMA_NAME:-$PROJECT_NAME}" \
        --temperature "${TEMP:-0.5}" \
        --repeat-penalty "${REP_PENALTY:-1.3}" \
        --num-predict "${NUM_PREDICT:-512}" \
        2>&1 | tee "$LOG_DIR/06_import_ollama.log"

    log_success "Ollama import complete"
    log_info "Model available as: ollama run ${OLLAMA_NAME:-$PROJECT_NAME}"
else
    log_info "Step 6: Import to Ollama - SKIPPED"
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
        --base-model "${BASE_MODEL:-deepseek-coder:6.7b-instruct}" \
        --finetuned-model "${OLLAMA_NAME:-$PROJECT_NAME}" \
        --num-tests "${NUM_TESTS:-10}" \
        --timeout "${TIMEOUT:-60}" \
        --output "$LOG_DIR/07_evaluation.json" \
        2>&1 | tee "$LOG_DIR/07_evaluation.log"

    log_success "Evaluation complete"
else
    log_info "Step 7: Evaluation - SKIPPED"
fi

# =============================================================================
# Summary
# =============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo -e "${GREEN}PIPELINE COMPLETE${NC}"
echo "============================================================"
echo ""
echo "Project: $PROJECT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Total time: ${MINUTES}m ${SECONDS}s"
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
echo "  ollama run ${OLLAMA_NAME:-$PROJECT_NAME}"
echo ""
