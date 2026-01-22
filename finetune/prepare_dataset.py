#!/usr/bin/env python3
"""Convert dataset.jsonl to formats suitable for fine-tuning."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

def convert_to_alpaca(input_path: Path, output_path: Path):
    """Convert to Alpaca format for unsloth/axolotl."""
    samples = []
    with input_path.open() as f:
        for line in f:
            item = json.loads(line)
            samples.append({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"]
            })

    with output_path.open("w") as f:
        json.dump(samples, f, indent=2)

    print(f"Converted {len(samples)} samples to {output_path}")


def convert_to_chatml(input_path: Path, output_path: Path):
    """Convert to ChatML format for chat fine-tuning."""
    with output_path.open("w") as f:
        with input_path.open() as inp:
            for line in inp:
                item = json.loads(line)

                # Build user message
                user_msg = item["instruction"]
                if item.get("input"):
                    user_msg += f"\n\nContext: {item['input']}"

                chat = {
                    "messages": [
                        {"role": "system", "content": "You are a HashiCorp infrastructure expert specializing in Terraform, Vault, Consul, Nomad, and Boundary."},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": item["output"]}
                    ]
                }
                f.write(json.dumps(chat) + "\n")

    print(f"Converted to ChatML format: {output_path}")


def convert_to_completion(input_path: Path, output_path: Path):
    """Convert to simple completion format for llama.cpp training."""
    with output_path.open("w") as f:
        with input_path.open() as inp:
            for line in inp:
                item = json.loads(line)

                text = f"""### Instruction:
{item["instruction"]}

### Response:
{item["output"]}
"""
                f.write(json.dumps({"text": text}) + "\n")

    print(f"Converted to completion format: {output_path}")


if __name__ == "__main__":
    input_path = DATA_DIR / "dataset.jsonl"

    # Create output directory
    output_dir = DATA_DIR / "training"
    output_dir.mkdir(exist_ok=True)

    # Convert to all formats
    convert_to_alpaca(input_path, output_dir / "alpaca.json")
    convert_to_chatml(input_path, output_dir / "chatml.jsonl")
    convert_to_completion(input_path, output_dir / "completion.jsonl")

    print(f"\nTraining data ready in {output_dir}/")
