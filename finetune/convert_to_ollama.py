#!/usr/bin/env python3
"""
Convert trained LoRA adapter to Ollama-compatible GGUF format.

Steps:
1. Load base model and LoRA adapter
2. Merge LoRA weights into base model
3. Save merged model
4. Convert to GGUF format using llama.cpp
5. Create Ollama Modelfile
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "terraform-expert"
MERGED_DIR = SCRIPT_DIR / "output" / "terraform-expert-merged"
GGUF_DIR = SCRIPT_DIR / "output" / "terraform-expert-gguf"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"

BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"


def merge_lora_adapter():
    """Merge LoRA adapter with base model using PEFT."""
    print("\n" + "=" * 60)
    print("Step 1: Merging LoRA adapter with base model")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_DIR}")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    print("✓ Merged model saved successfully")
    return MERGED_DIR


def convert_to_gguf(merged_dir: Path, quantization: str = "Q4_K_M"):
    """Convert merged model to GGUF format using llama.cpp."""
    print("\n" + "=" * 60)
    print(f"Step 2: Converting to GGUF format ({quantization})")
    print("=" * 60)

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    # Check for llama.cpp convert script
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try alternative location
        convert_script = LLAMA_CPP_DIR / "convert-hf-to-gguf.py"

    if not convert_script.exists():
        print(f"ERROR: llama.cpp convert script not found at {LLAMA_CPP_DIR}")
        print("Please ensure llama.cpp is installed at ~/llama.cpp")
        print("\nTo install:")
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("  cd ~/llama.cpp && uv pip install -r requirements.txt")
        sys.exit(1)

    output_file = GGUF_DIR / "terraform-expert-f16.gguf"

    print(f"Converting to F16 GGUF...")
    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(output_file),
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Conversion failed")
        print(result.stderr)
        sys.exit(1)

    print(f"✓ F16 GGUF saved to: {output_file}")

    # Quantize to Q4_K_M for smaller size
    if quantization != "f16":
        quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = LLAMA_CPP_DIR / "llama-quantize"

        if quantize_bin.exists():
            quantized_file = GGUF_DIR / f"terraform-expert-{quantization}.gguf"
            print(f"\nQuantizing to {quantization}...")

            cmd = [str(quantize_bin), str(output_file), str(quantized_file), quantization]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Quantized GGUF saved to: {quantized_file}")
                return quantized_file
            else:
                print(f"Warning: Quantization failed, using F16 version")
                print(result.stderr)
        else:
            print(f"Warning: llama-quantize not found, using F16 version")

    return output_file


def create_modelfile(gguf_path: Path):
    """Create Ollama Modelfile."""
    print("\n" + "=" * 60)
    print("Step 3: Creating Ollama Modelfile")
    print("=" * 60)

    modelfile_content = f'''# Terraform Expert - Fine-tuned on HashiCorp documentation
# Based on deepseek-coder-6.7b-instruct with LoRA fine-tuning

FROM {gguf_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "<|EOT|>"
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"

# System prompt for Terraform expertise
SYSTEM """You are a Terraform and Infrastructure as Code expert, trained on HashiCorp's official documentation. You provide accurate, detailed answers about:
- Terraform configuration language (HCL)
- Provider configurations (AWS, Azure, GCP, etc.)
- Resource and data source definitions
- State management and backends
- Modules and workspaces
- Best practices for infrastructure automation
- Troubleshooting common Terraform issues

Always provide code examples when appropriate and explain the reasoning behind your recommendations."""

# Chat template
TEMPLATE """{{{{ if .System }}}}### System:
{{{{ .System }}}}

{{{{ end }}}}### Instruction:
{{{{ .Prompt }}}}

### Response:
"""
'''

    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(modelfile_content)

    print(f"✓ Modelfile created at: {modelfile_path}")
    return modelfile_path


def import_to_ollama(modelfile_path: Path):
    """Import model into Ollama."""
    print("\n" + "=" * 60)
    print("Step 4: Importing into Ollama")
    print("=" * 60)

    model_name = "terraform-expert"

    print(f"Creating Ollama model: {model_name}")
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ Model '{model_name}' created successfully!")
        print(f"\nTo use the model:")
        print(f"  ollama run {model_name}")
        print(f'\nExample prompt:')
        print(f'  "How do I create an AWS S3 bucket with versioning enabled?"')
    else:
        print(f"\nERROR: Failed to create Ollama model")
        print(f"You can manually import with:")
        print(f"  cd {GGUF_DIR}")
        print(f"  ollama create {model_name} -f Modelfile")


def main():
    print("=" * 60)
    print("  Terraform Expert - LoRA to Ollama Converter")
    print("=" * 60)

    # Check if adapter exists
    adapter_path = OUTPUT_DIR / "adapter_model.safetensors"
    if not adapter_path.exists():
        print(f"ERROR: LoRA adapter not found at {adapter_path}")
        sys.exit(1)

    print(f"\nAdapter found: {adapter_path}")
    print(f"Size: {adapter_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Step 1: Merge LoRA with base model
    merged_dir = merge_lora_adapter()

    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(merged_dir, quantization="Q4_K_M")

    # Step 3: Create Modelfile
    modelfile_path = create_modelfile(gguf_path)

    # Step 4: Import to Ollama
    import_to_ollama(modelfile_path)

    print("\n" + "=" * 60)
    print("  Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Merged model: {merged_dir}")
    print(f"  GGUF file:    {gguf_path}")
    print(f"  Modelfile:    {modelfile_path}")


if __name__ == "__main__":
    main()
