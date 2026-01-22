# LLM Fine-Tuning Tutorials

Complete guides for fine-tuning Large Language Models using Axolotl on AMD ROCm and NVIDIA CUDA platforms.

## Tutorial Overview

| Tutorial | Description | When to Use |
|----------|-------------|-------------|
| [01-GPU-SETUP.md](./01-GPU-SETUP.md) | GPU environment setup | First-time setup |
| [02-STEP-BY-STEP-PIPELINE.md](./02-STEP-BY-STEP-PIPELINE.md) | Detailed pipeline walkthrough | Learning/debugging |
| [03-AUTOMATED-PIPELINE.md](./03-AUTOMATED-PIPELINE.md) | Single-script automation | Production use |

## Quick Start

### Prerequisites

- Linux (Ubuntu 22.04/24.04 recommended)
- Python 3.11+
- GPU: AMD MI50/MI100/MI200/MI300 or NVIDIA V100/A100/H100

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <your-repo>
cd llm_training_web_data

# Create virtual environment with uv
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install axolotl[flash-attn,deepspeed]
uv pip install transformers datasets accelerate peft
```

### GPU-Specific Setup

**For AMD ROCm (MI50/MI100/MI200):**
```bash
# Set environment for MI50
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0,1,2,3
```

**For NVIDIA CUDA:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Run Automated Pipeline

```bash
# Configure your training
vim finetune/pipeline_config.yaml

# Run the full pipeline
./scripts/run_pipeline.sh -y

# Or dry run first
./scripts/run_pipeline.sh --dry-run
```

## Hardware Requirements

| GPU | VRAM | Recommended Batch Size | Notes |
|-----|------|------------------------|-------|
| MI50 | 16GB | micro_batch_size: 1-2 | Requires HSA_OVERRIDE_GFX_VERSION |
| MI100 | 32GB | micro_batch_size: 2-4 | |
| MI200 | 64GB | micro_batch_size: 4-8 | |
| V100 | 16/32GB | micro_batch_size: 1-4 | |
| A100 | 40/80GB | micro_batch_size: 4-8 | |

## Pipeline Steps

1. **Data Preparation** - Convert raw data to Alpaca format
2. **Deduplication** - Remove duplicates and low-quality samples
3. **Training** - Fine-tune using LoRA with Axolotl
4. **Merge LoRA** - Combine adapter with base model
5. **Convert GGUF** - Create llama.cpp compatible format
6. **Import Ollama** - Make model available via Ollama
7. **Evaluation** - Compare against base model

## Key Configuration Options

### Hyperparameters for Stability

```yaml
training:
  learning_rate: 0.00005      # Lower = more stable
  weight_decay: 0.05          # Higher = more regularization
  max_grad_norm: 0.5          # Gradient clipping

lora:
  rank: 8                     # Lower = more regularization
  dropout: 0.1                # Higher = less overfitting
```

### Preventing Repetition

1. **Data**: Deduplicate training data
2. **Training**: Use lower learning rate, smaller LoRA rank
3. **Inference**: Set `repeat_penalty: 1.3` in Modelfile

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce `micro_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing

**Repetitive Output:**
- Check training data for duplicates
- Lower learning rate
- Increase `repeat_penalty` in inference

**Slow Training on MI50:**
- Expected ~6-8 minutes per step with gradient_accumulation_steps: 8
- Use sample_packing for efficiency

### Getting Help

- Check tutorial troubleshooting sections
- Review training logs in `finetune/logs/`
- Verify GPU setup with `rocm-smi` or `nvidia-smi`

## File Structure

```
llm_training_web_data/
├── data/
│   ├── raw/                    # Raw input data
│   └── training/               # Processed training data
├── finetune/
│   ├── output/                 # Trained models
│   ├── logs/                   # Training logs
│   ├── pipeline_config.yaml    # Pipeline configuration
│   └── axolotl_config*.yaml    # Axolotl configs
├── scripts/
│   ├── run_pipeline.sh         # Main pipeline script
│   ├── deduplicate_dataset.py  # Data cleaning
│   ├── merge_lora.py           # LoRA merge
│   ├── convert_to_gguf.py      # GGUF conversion
│   ├── create_ollama_model.py  # Ollama import
│   └── evaluate_model.py       # Model evaluation
└── docs/
    └── tutorials/              # This directory
```

## License

MIT License - See repository root for details.
