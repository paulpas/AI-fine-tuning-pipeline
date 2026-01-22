# Fine-tuning Guide for HashiCorp Terraform Expert Model

## Quick Start

### Step 1: Prepare Training Data
```bash
cd /home/paulpas/git/ideas/llm_training_web_data
uv run python finetune/prepare_dataset.py
```

This creates:
- `data/training/alpaca.json` - For Unsloth/Axolotl
- `data/training/chatml.jsonl` - For chat-style training
- `data/training/completion.jsonl` - For completion-style training

### Step 2: Install Unsloth
```bash
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --no-deps trl peft accelerate bitsandbytes
```

### Step 3: Fine-tune
```bash
uv run python finetune/train_unsloth.py \
    --base-model unsloth/Qwen2.5-7B-bnb-4bit \
    --epochs 3 \
    --batch-size 2
```

Recommended base models:
- `unsloth/Qwen2.5-7B-bnb-4bit` - Good balance of quality/speed
- `unsloth/Llama-3.2-3B-bnb-4bit` - Faster, smaller
- `unsloth/Mistral-7B-v0.3-bnb-4bit` - Strong general model

### Step 4: Import to Ollama
```bash
# After training, you'll have a GGUF file
cd finetune
ollama create terraform-expert -f Modelfile
```

### Step 5: Test
```bash
ollama run terraform-expert "How do I use Terraform workspaces?"
```

## Alternative: Use Existing Model with RAG

If you don't want to fine-tune, use RAG with your dataset:

```bash
# Create a simple expert model
ollama create terraform-expert -f - <<EOF
FROM gpt-oss:20b
SYSTEM "You are a HashiCorp Terraform expert. Answer based on official documentation."
PARAMETER temperature 0.7
EOF
```

Then use your `dataset.jsonl` as a knowledge base with a RAG system.

## Training Requirements

| Base Model | VRAM Required | Training Time (31k samples) |
|------------|---------------|----------------------------|
| 3B params  | 8GB           | ~2 hours                   |
| 7B params  | 16GB          | ~4 hours                   |
| 13B params | 24GB          | ~8 hours                   |

## Dataset Statistics
- **Total samples**: 31,958
- **Format**: Instruction/Response pairs
- **Source**: HashiCorp Developer documentation
- **Topics**: Terraform, Vault, Consul, Nomad, Boundary
