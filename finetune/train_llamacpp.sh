#!/bin/bash
# Fine-tune gpt-oss:20b using llama.cpp
#
# Prerequisites:
#   1. ROCm installed (for AMD MI50 GPUs)
#   2. llama.cpp built with HIP support
#
# Note: llama.cpp training works best with F32 models, quantized models may have issues

set -e

# ROCm environment for AMD MI50 (gfx906) - USE ALL 4 GPUs
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export GPU_DEVICE_ORDINAL=0,1,2,3
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="$ROCM_PATH/bin:$PATH"

# Paths
BASE_MODEL="/usr/share/ollama/.ollama/models/blobs/sha256-5ff0abeeac1d2dbdd5455c0b49ba3b29a9ce3c1fb181b2eef2e948689d55d046"  # deepseek-coder-v2:16b
TRAIN_DATA="/home/paulpas/git/ideas/llm_training_web_data/finetune/output/train.txt"
OUTPUT_DIR="/home/paulpas/git/ideas/llm_training_web_data/finetune/output"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

# Training parameters
BATCH_SIZE=512
CTX_SIZE=512
GPU_LAYERS=999  # Offload all layers to GPU
EPOCHS=2

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Fine-tuning deepseek-coder-v2:16b on AMD MI50 GPUs"
echo "=============================================="
echo "Base model: $BASE_MODEL"
echo "Training data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $HIP_VISIBLE_DEVICES"
echo "=============================================="

# Step 1: Convert training data to raw text format
echo "Converting training data to raw text..."
python3 << 'CONVERT_SCRIPT'
import json
from pathlib import Path

input_file = Path("/home/paulpas/git/ideas/llm_training_web_data/data/training/completion.jsonl")
output_file = Path("/home/paulpas/git/ideas/llm_training_web_data/finetune/output/train.txt")

with open(output_file, 'w') as out:
    with open(input_file) as inp:
        for line in inp:
            item = json.loads(line)
            out.write(item['text'] + '\n')

print(f"Training data written to {output_file}")
print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
CONVERT_SCRIPT

# Step 2: Run fine-tuning
echo ""
echo "Starting fine-tuning..."
echo "Command: llama-finetune --model $BASE_MODEL --file $TRAIN_DATA -ngl $GPU_LAYERS -c $CTX_SIZE -b $BATCH_SIZE -epochs $EPOCHS"
echo ""

"$LLAMA_CPP_DIR/build/bin/llama-finetune" \
    --model "$BASE_MODEL" \
    --file "$TRAIN_DATA" \
    -ngl $GPU_LAYERS \
    -c $CTX_SIZE \
    -b $BATCH_SIZE \
    -ub $BATCH_SIZE \
    -epochs $EPOCHS \
    -o "$OUTPUT_DIR/terraform-expert.gguf"

echo "=============================================="
echo "Training complete!"
echo "Output model: $OUTPUT_DIR/terraform-expert.gguf"
echo "=============================================="

# Step 3: Create Ollama Modelfile
cat > "$OUTPUT_DIR/Modelfile" << 'MODELFILE'
FROM ./terraform-expert.gguf

SYSTEM """You are a HashiCorp infrastructure expert specializing in Terraform, Vault, Consul, Nomad, and Boundary. Provide accurate answers based on official documentation."""

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
MODELFILE

echo ""
echo "To create the Ollama model, run:"
echo "  cd $OUTPUT_DIR"
echo "  ollama create terraform-expert -f Modelfile"
echo ""
echo "Then test with:"
echo "  ollama run terraform-expert 'How do I use Terraform workspaces?'"
