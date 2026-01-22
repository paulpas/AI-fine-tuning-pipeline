# Step-by-Step LLM Fine-Tuning Pipeline

This tutorial walks through each step of the fine-tuning pipeline individually, with verification at each stage.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step 1: Data Preparation](#step-1-data-preparation)
3. [Step 2: Data Deduplication & Cleaning](#step-2-data-deduplication--cleaning)
4. [Step 3: Configure Training](#step-3-configure-training)
5. [Step 4: Training Execution](#step-4-training-execution)
6. [Step 5: Model Conversion](#step-5-model-conversion)
7. [Step 6: Evaluation](#step-6-evaluation)
8. [Step 7: Deployment](#step-7-deployment)

---

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Data Collection │────▶│ Data Cleaning   │────▶│ Configuration   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐            │
│   Deployment    │◀────│   Evaluation    │◀───────────┘
└─────────────────┘     └─────────────────┘     ┌─────────────────┐
        ▲                       ▲               │    Training     │
        │                       │               └─────────────────┘
        │               ┌─────────────────┐            │
        └───────────────│   Conversion    │◀───────────┘
                        └─────────────────┘
```

---

## Step 1: Data Preparation

### 1.1 Understanding Data Formats

Axolotl supports multiple dataset formats. The most common is **Alpaca format**:

```json
[
  {
    "instruction": "Explain how to create an S3 bucket in Terraform",
    "input": "",
    "output": "To create an S3 bucket in Terraform, use the aws_s3_bucket resource:\n\n```hcl\nresource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket-name\"\n}\n```"
  }
]
```

Other supported formats:
- **ShareGPT**: Multi-turn conversations
- **Completion**: Simple text continuation
- **Chat**: OpenAI chat format

### 1.2 Create Data Directory Structure

```bash
mkdir -p data/raw
mkdir -p data/training
mkdir -p data/validation
```

### 1.3 Prepare Your Dataset

Create a script to format your data (`scripts/prepare_data.py`):

```python
#!/usr/bin/env python3
"""
Convert raw data to Alpaca format for training.
"""

import json
from pathlib import Path

def convert_to_alpaca(raw_data: list) -> list:
    """Convert raw data to Alpaca format."""
    alpaca_samples = []

    for item in raw_data:
        sample = {
            "instruction": item.get("question", item.get("prompt", "")),
            "input": item.get("context", ""),
            "output": item.get("answer", item.get("response", ""))
        }

        # Skip empty samples
        if sample["instruction"] and sample["output"]:
            alpaca_samples.append(sample)

    return alpaca_samples

def main():
    # Load raw data
    raw_path = Path("data/raw/dataset.json")
    output_path = Path("data/training/alpaca.json")

    with open(raw_path) as f:
        raw_data = json.load(f)

    # Convert to Alpaca format
    alpaca_data = convert_to_alpaca(raw_data)

    # Save formatted data
    with open(output_path, "w") as f:
        json.dump(alpaca_data, f, indent=2)

    print(f"Converted {len(raw_data)} raw samples to {len(alpaca_data)} training samples")

if __name__ == "__main__":
    main()
```

### 1.4 Verification: Data Format Check

```python
#!/usr/bin/env python3
"""
Verify dataset format and quality.
"""

import json
from pathlib import Path

def verify_dataset(path: str) -> bool:
    """Verify dataset is properly formatted."""
    with open(path) as f:
        data = json.load(f)

    print(f"Dataset: {path}")
    print(f"Total samples: {len(data)}")

    # Check required fields
    required_fields = ["instruction", "output"]
    missing_fields = 0

    for i, sample in enumerate(data[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Instruction: {sample.get('instruction', '')[:80]}...")
        print(f"  Output length: {len(sample.get('output', ''))}")

        for field in required_fields:
            if field not in sample or not sample[field]:
                missing_fields += 1

    # Statistics
    instruction_lengths = [len(s.get("instruction", "")) for s in data]
    output_lengths = [len(s.get("output", "")) for s in data]

    print(f"\n=== Statistics ===")
    print(f"Instruction length: min={min(instruction_lengths)}, max={max(instruction_lengths)}, avg={sum(instruction_lengths)//len(data)}")
    print(f"Output length: min={min(output_lengths)}, max={max(output_lengths)}, avg={sum(output_lengths)//len(data)}")

    if missing_fields == 0:
        print("\n✓ Dataset format verification: PASSED")
        return True
    else:
        print(f"\n✗ Dataset format verification: FAILED ({missing_fields} samples with missing fields)")
        return False

if __name__ == "__main__":
    verify_dataset("data/training/alpaca.json")
```

Run verification:
```bash
uv run python scripts/verify_data.py
```

---

## Step 2: Data Deduplication & Cleaning

Poor quality training data leads to repetitive or nonsensical model outputs. This step removes:
- Exact duplicates
- Near-duplicates
- Samples with repetitive patterns
- Very short outputs

### 2.1 Deduplication Script

Create `scripts/deduplicate_dataset.py`:

```python
#!/usr/bin/env python3
"""
Fast dataset deduplication using hashing.
Removes duplicates, short outputs, and repetitive patterns.
"""

import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def compute_hash(text: str) -> str:
    """Compute hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()

def count_repeated_phrases(text: str, min_len: int = 30) -> int:
    """Count repeated phrases in text (indicates repetitive content)."""
    text = normalize_text(text)

    # Split into overlapping chunks
    chunks = []
    for i in range(0, len(text) - min_len, min_len // 2):
        chunks.append(text[i:i + min_len])

    if not chunks:
        return 0

    # Count occurrences
    counts = defaultdict(int)
    for chunk in chunks:
        counts[chunk] += 1

    return max(counts.values()) if counts else 0

def deduplicate_dataset(
    input_path: str,
    output_path: str,
    min_output_length: int = 50,
    max_repetition: int = 3
) -> dict:
    """
    Deduplicate dataset and remove low-quality samples.

    Args:
        input_path: Path to input JSON dataset
        output_path: Path for output JSON dataset
        min_output_length: Minimum output length to keep
        max_repetition: Maximum allowed phrase repetitions

    Returns:
        Statistics dictionary
    """
    print(f"Loading dataset from: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    print(f"Original samples: {len(data):,}")

    # Track unique samples
    seen_instruction_hashes = set()
    seen_output_hashes = set()
    kept_samples = []

    stats = {
        "original": len(data),
        "duplicate_instruction": 0,
        "duplicate_output": 0,
        "repetitive_output": 0,
        "short_output": 0,
        "kept": 0,
    }

    for i, sample in enumerate(data):
        if i % 10000 == 0 and i > 0:
            print(f"Processing {i:,}/{len(data):,}...")

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        # Skip very short outputs
        if len(output.strip()) < min_output_length:
            stats["short_output"] += 1
            continue

        # Hash instruction+input combo
        instruction_hash = compute_hash(instruction + " " + input_text)

        # Skip duplicate instructions
        if instruction_hash in seen_instruction_hashes:
            stats["duplicate_instruction"] += 1
            continue

        # Hash output
        output_hash = compute_hash(output)

        # Skip duplicate outputs
        if output_hash in seen_output_hashes:
            stats["duplicate_output"] += 1
            continue

        # Check for repetitive patterns
        repetition_count = count_repeated_phrases(output)
        if repetition_count > max_repetition:
            stats["repetitive_output"] += 1
            continue

        # Keep this sample
        seen_instruction_hashes.add(instruction_hash)
        seen_output_hashes.add(output_hash)
        kept_samples.append(sample)
        stats["kept"] += 1

    # Save deduplicated dataset
    print(f"\nSaving deduplicated dataset to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(kept_samples, f, indent=2)

    # Print stats
    print("\n" + "=" * 50)
    print("Deduplication Statistics")
    print("=" * 50)
    print(f"Original samples:       {stats['original']:,}")
    print(f"Duplicate instructions: {stats['duplicate_instruction']:,}")
    print(f"Duplicate outputs:      {stats['duplicate_output']:,}")
    print(f"Repetitive outputs:     {stats['repetitive_output']:,}")
    print(f"Short outputs:          {stats['short_output']:,}")
    print(f"Kept samples:           {stats['kept']:,}")
    print(f"Reduction:              {(1 - stats['kept']/stats['original'])*100:.1f}%")
    print("=" * 50)

    return stats

if __name__ == "__main__":
    deduplicate_dataset(
        "data/training/alpaca.json",
        "data/training/alpaca_deduped.json"
    )
```

### 2.2 Run Deduplication

```bash
uv run python scripts/deduplicate_dataset.py
```

### 2.3 Verification: Check Cleaned Data

```bash
# Compare file sizes
ls -lh data/training/alpaca.json data/training/alpaca_deduped.json

# Quick quality check
python3 << 'EOF'
import json
import random

with open("data/training/alpaca_deduped.json") as f:
    data = json.load(f)

print(f"Cleaned dataset: {len(data)} samples")
print("\nRandom sample check:")
for sample in random.sample(data, min(3, len(data))):
    print(f"\nInstruction: {sample['instruction'][:80]}...")
    print(f"Output preview: {sample['output'][:100]}...")
    print(f"Output length: {len(sample['output'])} chars")
EOF
```

---

## Step 3: Configure Training

### 3.1 Understanding Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `learning_rate` | Controls weight update magnitude | 1e-5 to 2e-4 |
| `num_epochs` | Training iterations over dataset | 2-4 |
| `micro_batch_size` | Samples per GPU per step | 1-4 (memory dependent) |
| `gradient_accumulation_steps` | Steps before weight update | 4-16 |
| `lora_r` | LoRA rank (adapter capacity) | 8-64 |
| `lora_alpha` | LoRA scaling factor | Usually 2x lora_r |
| `lora_dropout` | Regularization dropout | 0.05-0.1 |

### 3.2 Create Training Configuration

Create `finetune/axolotl_config.yaml`:

```yaml
# Base model configuration
base_model: deepseek-ai/deepseek-coder-6.7b-instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

# Dataset configuration
datasets:
  - path: data/training/alpaca_deduped.json
    type: alpaca

# Output directory
output_dir: finetune/output/my-model

# LoRA configuration
adapter: lora
lora_r: 8                    # Lower rank = more regularization
lora_alpha: 16               # Usually 2x lora_r
lora_dropout: 0.1            # Higher dropout reduces overfitting
lora_target_linear: true
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training parameters
gradient_accumulation_steps: 8
micro_batch_size: 2
num_epochs: 4
learning_rate: 0.00005       # Lower LR = more stable training
weight_decay: 0.05           # Higher = more regularization
max_grad_norm: 0.5           # Clip gradients for stability
lr_scheduler: cosine
warmup_ratio: 0.1
optimizer: adamw_torch

# Data handling
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true
val_set_size: 0.1

# Evaluation and saving
eval_steps: 50
save_steps: 50
save_strategy: steps
eval_strategy: steps
save_total_limit: 5
load_best_model_at_end: true
metric_for_best_model: eval_loss
early_stopping_patience: 3

# Precision and memory
bf16: true
tf32: false
flash_attention: false       # Set true if supported

# Logging
logging_steps: 10
```

### 3.3 Configuration Validation

Create `scripts/validate_config.py`:

```python
#!/usr/bin/env python3
"""
Validate training configuration before starting.
"""

import yaml
from pathlib import Path

def validate_config(config_path: str) -> bool:
    """Validate training configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    errors = []
    warnings = []

    # Check required fields
    required = ["base_model", "output_dir", "datasets"]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Check dataset path exists
    if "datasets" in config:
        for ds in config["datasets"]:
            if not Path(ds["path"]).exists():
                errors.append(f"Dataset not found: {ds['path']}")

    # Check hyperparameter ranges
    lr = config.get("learning_rate", 0)
    if lr > 1e-3:
        warnings.append(f"Learning rate {lr} is very high, may cause instability")
    if lr < 1e-6:
        warnings.append(f"Learning rate {lr} is very low, training may be slow")

    lora_r = config.get("lora_r", 8)
    if lora_r > 64:
        warnings.append(f"LoRA rank {lora_r} is high, may increase memory usage")

    # Check output directory
    output_dir = Path(config.get("output_dir", ""))
    if output_dir.exists() and any(output_dir.iterdir()):
        warnings.append(f"Output directory {output_dir} is not empty")

    # Print results
    print("=" * 50)
    print("Configuration Validation")
    print("=" * 50)

    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("\n✅ Configuration is valid")

    print("=" * 50)

    # Summary
    print(f"\nModel: {config.get('base_model', 'Unknown')}")
    print(f"Dataset: {config.get('datasets', [{}])[0].get('path', 'Unknown')}")
    print(f"Output: {config.get('output_dir', 'Unknown')}")
    print(f"Learning rate: {config.get('learning_rate', 'Unknown')}")
    print(f"Epochs: {config.get('num_epochs', 'Unknown')}")
    print(f"LoRA rank: {config.get('lora_r', 'Unknown')}")

    return len(errors) == 0

if __name__ == "__main__":
    validate_config("finetune/axolotl_config.yaml")
```

Run validation:
```bash
uv run python scripts/validate_config.py
```

---

## Step 4: Training Execution

### 4.1 Set Up Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Set ROCm environment (for AMD GPUs)
source ~/.rocm_env  # Contains HSA_OVERRIDE_GFX_VERSION etc.

# Or for NVIDIA
# export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 4.2 Clear Previous Training

```bash
# Remove prepared data cache (forces re-tokenization)
rm -rf finetune/prepared_data

# Create output directory
mkdir -p finetune/output
```

### 4.3 Start Training

```bash
# Multi-GPU training with accelerate
accelerate launch -m axolotl.cli.train finetune/axolotl_config.yaml 2>&1 | tee finetune/training.log
```

Or for background training:

```bash
nohup accelerate launch -m axolotl.cli.train finetune/axolotl_config.yaml > finetune/training.log 2>&1 &
echo $! > finetune/training.pid
echo "Training started with PID $(cat finetune/training.pid)"
```

### 4.4 Monitor Training

```bash
# Watch training log
tail -f finetune/training.log

# Check GPU utilization
watch -n 2 rocm-smi --showuse  # For AMD
# watch -n 2 nvidia-smi        # For NVIDIA

# Check training progress
grep -E "(loss|step|epoch)" finetune/training.log | tail -20
```

### 4.5 Verification: Training Success

Create `scripts/verify_training.py`:

```python
#!/usr/bin/env python3
"""
Verify training completed successfully and model quality.
"""

import json
import re
from pathlib import Path

def parse_training_log(log_path: str) -> dict:
    """Parse training log for metrics."""
    metrics = {
        "initial_loss": None,
        "final_loss": None,
        "final_eval_loss": None,
        "total_steps": 0,
        "completed": False
    }

    with open(log_path) as f:
        content = f.read()

    # Find loss values
    loss_pattern = r"'loss':\s*([\d.]+)"
    losses = re.findall(loss_pattern, content)
    if losses:
        metrics["initial_loss"] = float(losses[0])
        metrics["final_loss"] = float(losses[-1])

    # Find eval loss
    eval_pattern = r"'eval_loss':\s*([\d.]+)"
    eval_losses = re.findall(eval_pattern, content)
    if eval_losses:
        metrics["final_eval_loss"] = float(eval_losses[-1])

    # Check completion
    if "Training complete" in content or "Saving final" in content:
        metrics["completed"] = True

    # Count steps
    step_pattern = r"(\d+)/\d+.*\[.*\]"
    steps = re.findall(step_pattern, content)
    if steps:
        metrics["total_steps"] = int(steps[-1])

    return metrics

def verify_model_files(output_dir: str) -> dict:
    """Verify model files exist."""
    output_path = Path(output_dir)

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    found = {}
    for f in required_files:
        path = output_path / f
        found[f] = path.exists()

    return found

def main():
    log_path = "finetune/training.log"
    output_dir = "finetune/output/my-model"

    print("=" * 60)
    print("Training Verification Report")
    print("=" * 60)

    # Parse training log
    metrics = parse_training_log(log_path)

    print("\n📊 Training Metrics:")
    print(f"  Initial loss: {metrics['initial_loss']}")
    print(f"  Final loss: {metrics['final_loss']}")
    print(f"  Final eval loss: {metrics['final_eval_loss']}")
    print(f"  Total steps: {metrics['total_steps']}")
    print(f"  Completed: {'✓' if metrics['completed'] else '✗'}")

    # Check loss improvement
    if metrics['initial_loss'] and metrics['final_loss']:
        improvement = (metrics['initial_loss'] - metrics['final_loss']) / metrics['initial_loss'] * 100
        print(f"  Loss improvement: {improvement:.1f}%")

        if improvement < 10:
            print("  ⚠️  Warning: Less than 10% loss improvement")
        elif improvement > 80:
            print("  ⚠️  Warning: Very high loss reduction, possible overfitting")
        else:
            print("  ✓ Loss improvement is in healthy range")

    # Verify model files
    print("\n📁 Model Files:")
    files = verify_model_files(output_dir)
    all_present = True
    for f, exists in files.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
        if not exists:
            all_present = False

    # Overall verdict
    print("\n" + "=" * 60)
    if metrics['completed'] and all_present and metrics['final_loss'] < metrics['initial_loss']:
        print("✅ TRAINING VERIFICATION: PASSED")
    else:
        print("❌ TRAINING VERIFICATION: FAILED")
        if not metrics['completed']:
            print("   - Training did not complete")
        if not all_present:
            print("   - Missing model files")
        if metrics['final_loss'] >= metrics['initial_loss']:
            print("   - Loss did not improve")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

Run verification:
```bash
uv run python scripts/verify_training.py
```

---

## Step 5: Model Conversion

Convert the trained LoRA adapter to formats usable by inference frameworks.

### 5.1 Merge LoRA with Base Model

Create `scripts/merge_lora.py`:

```python
#!/usr/bin/env python3
"""
Merge LoRA adapter with base model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def merge_lora(
    base_model: str,
    lora_path: str,
    output_path: str
):
    """Merge LoRA adapter with base model."""
    print(f"Loading base model: {base_model}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("✓ Model merged successfully")
    return output_path

if __name__ == "__main__":
    merge_lora(
        base_model="deepseek-ai/deepseek-coder-6.7b-instruct",
        lora_path="finetune/output/my-model",
        output_path="finetune/output/my-model-merged"
    )
```

### 5.2 Convert to GGUF Format

Create `scripts/convert_to_gguf.py`:

```python
#!/usr/bin/env python3
"""
Convert model to GGUF format for llama.cpp / Ollama.
"""

import subprocess
from pathlib import Path

def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m"
):
    """Convert model to GGUF format."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First convert to FP16 GGUF
    fp16_path = output_dir / "model-fp16.gguf"
    print(f"Converting to FP16 GGUF: {fp16_path}")

    # Use llama.cpp convert script
    # Assumes llama.cpp is cloned to ~/llama.cpp
    convert_cmd = [
        "python", f"{Path.home()}/llama.cpp/convert_hf_to_gguf.py",
        model_path,
        "--outfile", str(fp16_path),
        "--outtype", "f16"
    ]

    subprocess.run(convert_cmd, check=True)

    # Quantize
    quantized_path = output_dir / f"model-{quantization}.gguf"
    print(f"Quantizing to {quantization}: {quantized_path}")

    quantize_cmd = [
        f"{Path.home()}/llama.cpp/build/bin/llama-quantize",
        str(fp16_path),
        str(quantized_path),
        quantization
    ]

    subprocess.run(quantize_cmd, check=True)

    print(f"✓ GGUF conversion complete: {quantized_path}")
    return str(quantized_path)

if __name__ == "__main__":
    convert_to_gguf(
        model_path="finetune/output/my-model-merged",
        output_path="finetune/output/my-model-gguf",
        quantization="q4_k_m"
    )
```

### 5.3 Create Ollama Model

Create `scripts/create_ollama_model.py`:

```python
#!/usr/bin/env python3
"""
Import model into Ollama.
"""

import subprocess
from pathlib import Path

def create_modelfile(
    gguf_path: str,
    output_dir: str,
    model_name: str,
    system_prompt: str = ""
):
    """Create Ollama Modelfile."""
    modelfile_content = f'''FROM {gguf_path}

TEMPLATE """{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}<|end|>
"""

PARAMETER temperature 0.5
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.3
PARAMETER repeat_last_n 128
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
'''

    if system_prompt:
        modelfile_content += f'\nSYSTEM """{system_prompt}"""\n'

    modelfile_path = Path(output_dir) / "Modelfile"
    modelfile_path.write_text(modelfile_content)

    print(f"Created Modelfile at: {modelfile_path}")
    return str(modelfile_path)

def import_to_ollama(modelfile_path: str, model_name: str):
    """Import model to Ollama."""
    print(f"Importing to Ollama as: {model_name}")

    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    subprocess.run(cmd, check=True)

    print(f"✓ Model imported to Ollama: {model_name}")
    print(f"  Run with: ollama run {model_name}")

if __name__ == "__main__":
    create_modelfile(
        gguf_path="finetune/output/my-model-gguf/model-q4_k_m.gguf",
        output_dir="finetune/output/my-model-gguf",
        model_name="my-model"
    )

    import_to_ollama(
        "finetune/output/my-model-gguf/Modelfile",
        "my-model"
    )
```

### 5.4 Verification: Test Converted Model

```bash
# Test with Ollama
ollama run my-model "Write a hello world program in Python"

# Check model info
ollama show my-model
```

---

## Step 6: Evaluation

### 6.1 Create Evaluation Script

Create `scripts/evaluate_model.py`:

```python
#!/usr/bin/env python3
"""
Evaluate fine-tuned model against base model.
"""

import subprocess
import json
import re
from typing import Optional

# Evaluation questions
EVAL_QUESTIONS = [
    {
        "id": "basic_syntax",
        "question": "Write a Python function to calculate factorial",
        "expected_keywords": ["def", "factorial", "return", "if", "else"]
    },
    {
        "id": "domain_specific",
        "question": "How do I create a variable in Terraform?",
        "expected_keywords": ["variable", "type", "default", "description"]
    },
    # Add more domain-specific questions
]

def run_ollama(model: str, prompt: str, timeout: int = 60) -> Optional[str]:
    """Run Ollama model and get response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_keyword_coverage(response: str, keywords: list) -> float:
    """Calculate what percentage of expected keywords are present."""
    if not response:
        return 0.0

    response_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    return found / len(keywords) * 100

def evaluate_models(base_model: str, finetuned_model: str):
    """Compare base and fine-tuned models."""
    results = []

    for q in EVAL_QUESTIONS:
        print(f"\nEvaluating: {q['id']}")
        print(f"Question: {q['question'][:50]}...")

        # Test base model
        base_response = run_ollama(base_model, q['question'])
        base_score = calculate_keyword_coverage(
            base_response, q['expected_keywords']
        )

        # Test fine-tuned model
        ft_response = run_ollama(finetuned_model, q['question'])
        ft_score = calculate_keyword_coverage(
            ft_response, q['expected_keywords']
        )

        results.append({
            "question_id": q['id'],
            "base_score": base_score,
            "finetuned_score": ft_score,
            "improvement": ft_score - base_score
        })

        print(f"  Base model score: {base_score:.1f}%")
        print(f"  Fine-tuned score: {ft_score:.1f}%")
        print(f"  Improvement: {ft_score - base_score:+.1f}%")

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    avg_base = sum(r['base_score'] for r in results) / len(results)
    avg_ft = sum(r['finetuned_score'] for r in results) / len(results)
    avg_improvement = avg_ft - avg_base

    print(f"Average base model score: {avg_base:.1f}%")
    print(f"Average fine-tuned score: {avg_ft:.1f}%")
    print(f"Average improvement: {avg_improvement:+.1f}%")

    if avg_improvement > 10:
        print("\n✅ EVALUATION: Fine-tuning shows significant improvement")
    elif avg_improvement > 0:
        print("\n⚠️  EVALUATION: Fine-tuning shows modest improvement")
    else:
        print("\n❌ EVALUATION: Fine-tuning did not improve performance")

    # Save results
    with open("finetune/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: finetune/evaluation_results.json")

if __name__ == "__main__":
    evaluate_models(
        base_model="deepseek-coder:6.7b-instruct",
        finetuned_model="my-model"
    )
```

### 6.2 Run Evaluation

```bash
uv run python scripts/evaluate_model.py
```

### 6.3 Check for Repetition Issues

Create `scripts/check_repetition.py`:

```python
#!/usr/bin/env python3
"""
Check for repetition issues in model output.
"""

import subprocess
import re

def check_repetition(model: str, num_tests: int = 5):
    """Test model for repetitive output."""
    test_prompts = [
        "Explain how to use Python lists",
        "What is a variable in programming?",
        "How do I write a loop?",
    ]

    issues = 0

    for prompt in test_prompts[:num_tests]:
        print(f"\nTesting: {prompt[:40]}...")

        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        response = result.stdout

        # Check for repeated phrases
        sentences = re.split(r'[.!?]', response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]

        unique_sentences = set(sentences)
        if len(sentences) > 0:
            repetition_ratio = 1 - len(unique_sentences) / len(sentences)

            if repetition_ratio > 0.3:
                print(f"  ⚠️  High repetition detected: {repetition_ratio*100:.1f}%")
                issues += 1
            else:
                print(f"  ✓ Repetition ratio: {repetition_ratio*100:.1f}%")

    print("\n" + "=" * 50)
    if issues == 0:
        print("✅ No significant repetition issues detected")
    else:
        print(f"⚠️  {issues} tests showed repetition issues")
        print("Consider:")
        print("  - Increasing repetition_penalty in Modelfile")
        print("  - Retraining with lower learning rate")
        print("  - Checking for repetitive patterns in training data")

if __name__ == "__main__":
    check_repetition("my-model")
```

---

## Step 7: Deployment

### 7.1 Production Modelfile

Create optimized Modelfile for production:

```bash
cat > finetune/output/my-model-gguf/Modelfile.production << 'EOF'
FROM ./model-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""

# Optimized parameters for production
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 1024
PARAMETER repeat_penalty 1.2
PARAMETER repeat_last_n 64

# Stop sequences
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"

# System prompt
SYSTEM """You are a helpful coding assistant. Provide clear, accurate, and concise responses."""
EOF
```

### 7.2 Deploy to Ollama

```bash
# Create production model
ollama create my-model-prod -f finetune/output/my-model-gguf/Modelfile.production

# Verify deployment
ollama list | grep my-model
```

### 7.3 API Usage Example

```python
#!/usr/bin/env python3
"""
Example: Using the deployed model via Ollama API.
"""

import requests

def query_model(prompt: str, model: str = "my-model-prod"):
    """Query model via Ollama API."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"API error: {response.status_code}")

if __name__ == "__main__":
    result = query_model("Write a hello world in Python")
    print(result)
```

---

## Summary Checklist

Use this checklist to verify each step:

- [ ] **Data Preparation**
  - [ ] Data converted to Alpaca format
  - [ ] Data format verified

- [ ] **Data Cleaning**
  - [ ] Deduplication completed
  - [ ] Removed short outputs
  - [ ] Removed repetitive samples

- [ ] **Configuration**
  - [ ] Config file created
  - [ ] Hyperparameters validated
  - [ ] Dataset path correct

- [ ] **Training**
  - [ ] Training completed without errors
  - [ ] Loss improved from initial to final
  - [ ] Model files saved

- [ ] **Conversion**
  - [ ] LoRA merged with base model
  - [ ] GGUF file created
  - [ ] Ollama model imported

- [ ] **Evaluation**
  - [ ] Model responds correctly
  - [ ] No repetition issues
  - [ ] Performance improved over base

- [ ] **Deployment**
  - [ ] Production model created
  - [ ] API accessible

---

## Next Steps

- For automated execution of all steps, see [03-AUTOMATED-PIPELINE.md](./03-AUTOMATED-PIPELINE.md)
- For troubleshooting common issues, see [04-TROUBLESHOOTING.md](./04-TROUBLESHOOTING.md)
