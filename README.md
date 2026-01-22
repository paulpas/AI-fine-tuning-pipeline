# LLM Fine-Tuning Pipeline

> **A production-ready, modularized pipeline for fine-tuning large language models on custom datasets with automatic GPU utilization and turn-key deployment.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

## Overview

This project provides a **complete, end-to-end pipeline** for fine-tuning large language models using LoRA (Low-Rank Adaptation) on custom datasets. It's designed for:

- **Easy to use**: Single YAML configuration file controls everything
- **Hardware-optimized**: Automatically detects and utilizes all available GPUs
- **Production-ready**: Modularized, extensible, fault-tolerant design
- **Fully automated**: From data collection to model deployment
- **Well-documented**: Professional-grade reference guide (AGENTS.md)

## Key Features

✨ **Six-Stage Pipeline**
- **Collect**: Clone GitHub repositories automatically
- **Extract**: Parse Python code, documentation, and function signatures
- **Combine**: Merge datasets from multiple sources
- **Deduplicate**: Remove duplicates and low-quality examples
- **Train**: Fine-tune models with distributed GPU training (DDP)
- **Export**: Merge adapters, quantize to GGUF, deploy to Ollama

🚀 **Hardware Acceleration**
- Automatic multi-GPU detection and utilization (DDP)
- Support for AMD ROCm and NVIDIA CUDA
- Intelligent memory management
- Real-time GPU monitoring and health checks

🔧 **Configuration-Driven**
- Single `config/pipeline_config.yaml` file
- Environment variable overrides for CI/CD
- Profile support (test, production, custom)
- Model-specific auto-detection

💾 **Data Handling**
- Python AST parsing for code extraction
- RST and Markdown documentation support
- Intelligent deduplication and quality filtering
- Configurable output formats (Alpaca, ShareGPT)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/paulpas/ideas.git
cd ideas/llm_training_web_data

# Install uv (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (via PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

> **Why uv?** It's 10-100x faster than pip with better dependency resolution and lock file support.

### Configuration

Edit `config/pipeline_config.yaml`:

```yaml
pipeline:
  name: my-expert-model
  description: Custom domain expert model

git_sources:
  - name: my-repo
    url: https://github.com/user/repo
    enabled: true

training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  num_gpus: 4           # Auto-uses all available
  num_epochs: 3
```

### Run the Pipeline

```bash
# Complete pipeline: collect → extract → combine → dedupe → train → export
python -m pipeline.runner --config config/pipeline_config.yaml

# Or run specific stages
python -m pipeline.runner --stage train,export

# Use a profile for different configurations
python -m pipeline.runner --profile production
```

### Use Your Trained Model

```bash
# Run with Ollama
ollama run my-expert-model

# Or in your application
from ollama import Client
client = Client()
response = client.generate(model='my-expert-model', prompt='Your prompt here')
print(response)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  LLM FINE-TUNING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Collection & Processing        Model Training & Export     │
│  ════════════════════════════════════════════════════════════   │
│                                                                  │
│  1. Collect        Clone repositories                            │
│     ↓                                                            │
│  2. Extract        Parse code & documentation                   │
│     ↓                                                            │
│  3. Combine        Merge datasets                               │
│     ↓                                                            │
│  4. Deduplicate    Remove duplicates & low quality              │
│     ↓                                                            │
│  5. Train          Fine-tune with all available GPUs (DDP)      │
│     ↓                                                            │
│  6. Export         Merge LoRA → GGUF → Ollama Model             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Supported Models

| Model Family | Example | Auto-Detected |
|-------------|---------|---------------|
| **DeepSeek** | DeepSeek-R1-Distill-Qwen-1.5B | ✅ |
| **CodeGemma** | google/codegemma-2b | ✅ |
| **Qwen** | Qwen2-7B | ✅ |
| **LLaMA/Mistral** | mistral-ai/Mistral-7B | ✅ |
| **Phi** | microsoft/phi-2 | ✅ |

The pipeline auto-detects your model's chat template and applies appropriate configurations. Override manually if needed.

## Performance

### Benchmark Results

| Model | GPUs | Batch | Time | VRAM/GPU |
|-------|------|-------|------|----------|
| DeepSeek-1.5B | 1 | 4 | 8-12h | 8GB |
| DeepSeek-1.5B | 4 | 4 | 2-3h | 4GB |
| CodeGemma-2B | 1 | 4 | 6-10h | 6GB |
| Mistral-7B | 4 | 2 | 12-24h | 12GB |

### Hardware Requirements

**Minimum:**
- GPU: Any modern GPU with 6GB+ VRAM
- RAM: 16GB system memory
- Storage: 50GB (for model + datasets)

**Recommended (for 4-GPU training):**
- GPUs: 4x A100/H100 (NVIDIA) or MI300 (AMD)
- RAM: 128GB system memory
- Storage: 500GB (for checkpoints + models)

## Configuration Reference

### Minimal Configuration

```yaml
training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  num_epochs: 3
```

### Advanced Configuration

```yaml
training:
  base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  num_gpus: 4

  lora:
    r: 16
    alpha: 32
    dropout: 0.1

  hyperparameters:
    learning_rate: 3.0e-05
    micro_batch_size: 4
    gradient_accumulation_steps: 2
    sequence_len: 2048
    num_epochs: 3

  early_stopping:
    enabled: true
    patience: 10

export:
  gguf:
    quantization: q4_k_m
  ollama:
    temperature: 0.7
    repeat_penalty: 1.1
```

See **[AGENTS.md](AGENTS.md)** for the complete configuration reference and advanced options.

## Troubleshooting

### GPU Issues

**Problem:** GPUs not detected
**Solution:**
```bash
# AMD ROCm
rocm-smi  # Verify GPU detection
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Adjust for your GPU

# NVIDIA CUDA
nvidia-smi  # Verify GPU detection
```

**Problem:** Out of memory errors
**Solution:** Reduce `micro_batch_size` or `sequence_len` in config:
```yaml
training:
  hyperparameters:
    micro_batch_size: 1
    sequence_len: 512
```

### Training Issues

**Problem:** Loss not decreasing
**Solutions:**
- Increase learning rate: `0.0001`
- Increase epochs: `num_epochs: 5`
- Check data quality: `head data/training/deduped.json`

**Problem:** Training crashes mid-way
**Solutions:**
- The pipeline automatically resumes from the latest checkpoint
- Reduce batch size or sequence length
- Check GPU temperatures: `rocm-smi` or `nvidia-smi`

For more troubleshooting, see [AGENTS.md](AGENTS.md#troubleshooting).

## Project Structure

```
llm_training_web_data/
├── README.md                          # This file
├── AGENTS.md                          # Complete technical reference
├── TRAINING_STATUS.txt                # Current training status
│
├── config/
│   └── pipeline_config.yaml           # Main configuration file
│
├── data/
│   └── training/
│       ├── *_extracted.json           # Extracted training data
│       ├── combined.json              # Merged datasets
│       └── deduped.json               # Final deduplicated data
│
├── finetune/
│   ├── axolotl_config_*.yaml          # Generated training configs
│   └── output/
│       └── {model_name}/
│           ├── checkpoint-*/          # Training checkpoints
│           ├── merged/                # Merged LoRA + base model
│           ├── gguf/                  # Quantized GGUF files
│           └── *.log                  # Training logs
│
├── pipeline/                          # Core pipeline module
│   ├── runner.py                      # Main orchestrator
│   ├── config_loader.py               # Configuration management
│   ├── data_extractor.py              # Extract training data
│   ├── data_processor.py              # Dedup & combine
│   └── model_exporter.py              # Merge, GGUF, Ollama export
│
├── repos/                             # Cloned repositories
├── requirements.txt                   # Python dependencies
└── .venv/                             # Virtual environment
```

## Modular Design

Each pipeline stage is **independent and reusable**:

```python
# Use individual stages programmatically
from pipeline.runner import stage_collect, stage_train, stage_export

config = load_config("config/pipeline_config.yaml")

# Run only what you need
stage_collect(config)
stage_train(config)
stage_export(config)
```

Extend with custom stages:

```python
def stage_custom(config):
    # Your implementation
    return StageResult(stage="custom", success=True, ...)

STAGES["custom"] = stage_custom
```

## Environment Variables

Override configuration via environment variables:

```bash
export PIPELINE_TRAINING_NUM_GPUS=2
export PIPELINE_TRAINING_BASE_MODEL="mistral-ai/Mistral-7B"
export PIPELINE_TRAINING_LEARNING_RATE=0.0001

python -m pipeline.runner
```

## Development

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Python 3.10+
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Test changes thoroughly

### Adding New Models

To support a new model family:

1. Add model detection in `pipeline/config_loader.py`
2. Configure chat template in `config/pipeline_config.yaml`
3. Test with `--profile test`
4. Update [AGENTS.md](AGENTS.md#supported-models)

## Performance Tips

1. **Use 4-GPU training for best speed** (if available)
2. **Start with smaller sequence length** if memory-constrained
3. **Increase `gradient_accumulation_steps`** to maintain effective batch size
4. **Enable `flash_attention`** for faster training (if supported)
5. **Use `q4_k_m` quantization** for GGUF (good quality/speed tradeoff)

## Monitoring

### During Training

```bash
# Watch real-time progress
tail -f finetune/output/*/training.log

# Check GPU status
watch -n 1 rocm-smi      # AMD
watch -n 1 nvidia-smi    # NVIDIA

# Check loss trend
grep "loss" training.log | tail -20
```

### After Training

```bash
# List checkpoints
ls finetune/output/*/checkpoint-*/

# Check final model
ls finetune/output/*/ollama/
```

## Community & Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: See [AGENTS.md](AGENTS.md) for detailed reference
- **Examples**: Check `config/pipeline_config.yaml` for configuration examples

## Roadmap

- [ ] Web UI for configuration
- [ ] Model evaluation metrics dashboard
- [ ] Multi-dataset ranking & selection
- [ ] Distributed training across multiple machines
- [ ] Model compression techniques (pruning, distillation)
- [ ] A/B testing framework for hyperparameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{llm_finetuning_pipeline,
  title={LLM Fine-Tuning Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/paulpas/ideas}
}
```

## Acknowledgments

Built with:
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Fine-tuning framework
- [Transformers](https://huggingface.co/transformers/) - Model loading & inference
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Ollama](https://ollama.ai/) - Model deployment

---

**Ready to train your custom LLM?** Start with the [Quick Start](#quick-start) section or read [AGENTS.md](AGENTS.md) for complete details.

**Current Status**: Training in progress • 4-GPU • Stable • Auto-export enabled ✨
