#!/usr/bin/env python3
"""
Merge LoRA adapter with base model.
"""

import torch
import argparse
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

    print("Model merged successfully")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base-model", required=True, help="Base model name/path")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output path for merged model")

    args = parser.parse_args()

    merge_lora(args.base_model, args.lora_path, args.output)
