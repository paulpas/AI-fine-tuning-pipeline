#!/usr/bin/env python3
"""
Convert model to GGUF format for llama.cpp / Ollama.
"""

import subprocess
import argparse
from pathlib import Path
import os


def find_llama_cpp():
    """Find llama.cpp installation."""
    # Check common locations
    locations = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
    ]

    for loc in locations:
        if (loc / "convert_hf_to_gguf.py").exists():
            return loc

    # Check if in PATH
    try:
        result = subprocess.run(
            ["which", "llama-quantize"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).parent.parent
    except Exception:
        pass

    return None


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m"
):
    """Convert model to GGUF format."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    llama_cpp_path = find_llama_cpp()

    if llama_cpp_path is None:
        print("ERROR: llama.cpp not found.")
        print("Please install llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("  cd ~/llama.cpp && make -j")
        return None

    print(f"Using llama.cpp from: {llama_cpp_path}")

    # First convert to FP16 GGUF
    fp16_path = output_dir / "model-fp16.gguf"
    print(f"Converting to FP16 GGUF: {fp16_path}")

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    convert_cmd = [
        "python3", str(convert_script),
        model_path,
        "--outfile", str(fp16_path),
        "--outtype", "f16"
    ]

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion error: {result.stderr}")
        return None

    print("FP16 conversion complete")

    # Quantize
    quantized_path = output_dir / f"model-{quantization}.gguf"
    print(f"Quantizing to {quantization}: {quantized_path}")

    # Find quantize binary
    quantize_bin = None
    for name in ["llama-quantize", "quantize"]:
        for subdir in ["build/bin", "build", ""]:
            path = llama_cpp_path / subdir / name
            if path.exists():
                quantize_bin = path
                break
        if quantize_bin:
            break

    if quantize_bin is None:
        print("WARNING: Quantize binary not found, keeping FP16")
        return str(fp16_path)

    quantize_cmd = [
        str(quantize_bin),
        str(fp16_path),
        str(quantized_path),
        quantization
    ]

    result = subprocess.run(quantize_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Quantization error: {result.stderr}")
        print("Keeping FP16 model")
        return str(fp16_path)

    # Remove FP16 to save space
    fp16_path.unlink()

    print(f"GGUF conversion complete: {quantized_path}")
    return str(quantized_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("--model-path", required=True, help="Path to merged model")
    parser.add_argument("--output-path", required=True, help="Output directory for GGUF")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization type")

    args = parser.parse_args()

    convert_to_gguf(args.model_path, args.output_path, args.quantization)
