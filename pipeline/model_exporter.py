"""
Unified Model Exporter

Handles the complete model export pipeline:
1. Merge LoRA adapter with base model
2. Convert to GGUF format
3. Create and import Ollama model

Replaces: scripts/merge_lora.py, scripts/convert_to_gguf.py, scripts/create_ollama_model.py

Usage:
    from pipeline.model_exporter import export_model
    from pipeline.config_loader import load_config

    config = load_config()
    export_model(
        config=config,
        checkpoint_path="/path/to/checkpoint",
    )
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of model export operation."""
    success: bool
    merged_path: Optional[Path] = None
    gguf_path: Optional[Path] = None
    ollama_model: Optional[str] = None
    error: Optional[str] = None


def find_llama_cpp(search_paths: list = None) -> Optional[Path]:
    """
    Find llama.cpp installation.

    Args:
        search_paths: Additional paths to search

    Returns:
        Path to llama.cpp directory or None
    """
    default_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
    ]

    if search_paths:
        default_paths = [Path(p) for p in search_paths] + default_paths

    for loc in default_paths:
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


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in an output directory.

    Args:
        output_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by checkpoint number
    def get_checkpoint_num(p: Path) -> int:
        try:
            return int(p.name.split("-")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_checkpoint_num, reverse=True)
    return checkpoints[0]


def merge_lora(
    base_model: str,
    lora_path: Path,
    output_path: Path,
    torch_dtype: str = "float16",
    device_map: str = "cpu",
) -> Tuple[bool, Optional[str]]:
    """
    Merge LoRA adapter with base model.

    Args:
        base_model: Base model name/path
        lora_path: Path to LoRA adapter
        output_path: Output path for merged model
        torch_dtype: Torch data type
        device_map: Device map for loading

    Returns:
        Tuple of (success, error_message)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        log.info(f"Loading base model: {base_model}")

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        log.info(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

        log.info("Merging weights...")
        model = model.merge_and_unload()

        log.info(f"Saving merged model to: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        log.info("Model merged successfully")
        return True, None

    except Exception as e:
        log.error(f"Merge failed: {e}")
        return False, str(e)


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m",
    llama_cpp_path: Optional[Path] = None,
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Convert model to GGUF format.

    Args:
        model_path: Path to merged model
        output_path: Output directory for GGUF
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16)
        llama_cpp_path: Path to llama.cpp installation

    Returns:
        Tuple of (success, gguf_path, error_message)
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Find llama.cpp
    llama_cpp = llama_cpp_path or find_llama_cpp()

    if llama_cpp is None:
        error = (
            "llama.cpp not found. Please install:\n"
            "  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "  cd ~/llama.cpp && make -j"
        )
        log.error(error)
        return False, None, error

    log.info(f"Using llama.cpp from: {llama_cpp}")

    # First convert to FP16 GGUF
    fp16_path = output_path / "model-fp16.gguf"
    log.info(f"Converting to FP16 GGUF: {fp16_path}")

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    convert_cmd = [
        "python3", str(convert_script),
        str(model_path),
        "--outfile", str(fp16_path),
        "--outtype", "f16"
    ]

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error = f"Conversion error: {result.stderr}"
        log.error(error)
        return False, None, error

    log.info("FP16 conversion complete")

    # If f16 requested, we're done
    if quantization == "f16":
        return True, fp16_path, None

    # Quantize
    quantized_path = output_path / f"model-{quantization}.gguf"
    log.info(f"Quantizing to {quantization}: {quantized_path}")

    # Find quantize binary
    quantize_bin = None
    for name in ["llama-quantize", "quantize"]:
        for subdir in ["build/bin", "build", ""]:
            path = llama_cpp / subdir / name
            if path.exists():
                quantize_bin = path
                break
        if quantize_bin:
            break

    if quantize_bin is None:
        log.warning("Quantize binary not found, keeping FP16")
        return True, fp16_path, None

    quantize_cmd = [
        str(quantize_bin),
        str(fp16_path),
        str(quantized_path),
        quantization
    ]

    result = subprocess.run(quantize_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(f"Quantization failed: {result.stderr}")
        log.warning("Keeping FP16 model")
        return True, fp16_path, None

    # Remove FP16 to save space
    try:
        fp16_path.unlink()
    except Exception:
        pass

    log.info(f"GGUF conversion complete: {quantized_path}")
    return True, quantized_path, None


def detect_model_family(base_model: str) -> str:
    """
    Detect model family from HuggingFace model ID.

    Args:
        base_model: HuggingFace model ID or path

    Returns:
        Model family: deepseek, gemma, qwen, llama, phi, or generic
    """
    model_lower = base_model.lower()

    if "deepseek" in model_lower:
        return "deepseek"
    elif "gemma" in model_lower or "codegemma" in model_lower:
        return "gemma"
    elif "qwen" in model_lower:
        return "qwen"
    elif "llama" in model_lower or "mistral" in model_lower:
        return "llama"
    elif "phi" in model_lower:
        return "phi"
    else:
        return "phi"  # Default to phi template (works for most models)


def get_ollama_template(model_family: str) -> Tuple[str, list]:
    """
    Get Ollama TEMPLATE and stop tokens for a model family.

    Args:
        model_family: Model family name

    Returns:
        Tuple of (template_string, stop_tokens_list)
    """
    templates = {
        "deepseek": (
            '''TEMPLATE """{{ if .System }}<|begin▁of▁sentence|>{{ .System }}
{{ end }}<|User|>{{ .Prompt }}
<|Assistant|>{{ .Response }}<|end▁of▁sentence|>"""''',
            ["<|end▁of▁sentence|>", "<|User|>", "<|Assistant|>"]
        ),
        "gemma": (
            '''TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"""''',
            ["<end_of_turn>", "<start_of_turn>"]
        ),
        "qwen": (
            '''TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""''',
            ["<|im_end|>", "<|im_start|>"]
        ),
        "llama": (
            '''TEMPLATE """{{ if .System }}[INST] <<SYS>>
{{ .System }}
<</SYS>>

{{ .Prompt }} [/INST] {{ else }}[INST] {{ .Prompt }} [/INST] {{ end }}{{ .Response }}"""''',
            ["[INST]", "[/INST]", "</s>"]
        ),
        "phi": (
            '''TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ .Response }}<|end|>"""''',
            ["<|end|>", "<|user|>", "<|assistant|>"]
        ),
    }

    return templates.get(model_family, templates["phi"])


def create_ollama_model(
    gguf_path: Path,
    model_name: str,
    temperature: float = 0.5,
    repeat_penalty: float = 1.3,
    num_predict: int = 512,
    system_prompt: str = "",
    base_model: str = "",
    chat_template: str = "auto",
) -> Tuple[bool, Optional[str]]:
    """
    Create and import Ollama model.

    Args:
        gguf_path: Path to GGUF file
        model_name: Name for Ollama model
        temperature: Generation temperature
        repeat_penalty: Repeat penalty
        num_predict: Max tokens to predict
        system_prompt: System prompt
        base_model: Base model name for template detection
        chat_template: Chat template type (auto, deepseek, gemma, qwen, llama, phi)

    Returns:
        Tuple of (success, error_message)
    """
    # Make path absolute
    gguf_path = gguf_path.absolute()

    # Detect model family for template
    if chat_template == "auto" and base_model:
        model_family = detect_model_family(base_model)
        log.info(f"Auto-detected model family: {model_family}")
    elif chat_template != "auto":
        model_family = chat_template
    else:
        model_family = "phi"

    # Get template and stop tokens
    template_str, stop_tokens = get_ollama_template(model_family)

    # Build Modelfile
    modelfile_content = f'''FROM {gguf_path}

{template_str}

PARAMETER temperature {temperature}
PARAMETER num_predict {num_predict}
PARAMETER repeat_penalty {repeat_penalty}
PARAMETER repeat_last_n 128
'''

    # Add stop tokens
    for stop in stop_tokens:
        modelfile_content += f'PARAMETER stop "{stop}"\n'

    if system_prompt:
        modelfile_content += f'\nSYSTEM """{system_prompt}"""\n'

    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    log.info(f"Created Modelfile at: {modelfile_path}")
    log.info(f"Using {model_family} chat template")

    # Import to Ollama
    log.info(f"Importing to Ollama as: {model_name}")
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error = f"Ollama import failed: {result.stderr}"
        log.error(error)
        return False, error

    log.info(f"Model imported to Ollama: {model_name}")
    log.info(f"Run with: ollama run {model_name}")
    return True, None


def export_model(
    config: "PipelineConfig",
    checkpoint_path: Optional[Path] = None,
    skip_merge: bool = False,
    skip_gguf: bool = False,
    skip_ollama: bool = False,
) -> ExportResult:
    """
    Run the complete model export pipeline.

    Args:
        config: Pipeline configuration
        checkpoint_path: Path to checkpoint (auto-detected if not specified)
        skip_merge: Skip LoRA merge step
        skip_gguf: Skip GGUF conversion step
        skip_ollama: Skip Ollama import step

    Returns:
        ExportResult with paths and status
    """
    result = ExportResult(success=False)

    # Find checkpoint if not specified
    if checkpoint_path is None:
        output_dir = config.paths.output.checkpoints / config.get_output_name()
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path is None:
            result.error = f"No checkpoint found in {output_dir}"
            return result

    log.info(f"Using checkpoint: {checkpoint_path}")

    # Step 1: Merge LoRA
    merged_path = config.paths.output.merged / config.get_output_name()

    if not skip_merge:
        success, error = merge_lora(
            base_model=config.training.base_model,
            lora_path=checkpoint_path,
            output_path=merged_path,
            torch_dtype=config.export.merge.torch_dtype,
            device_map=config.export.merge.device_map,
        )
        if not success:
            result.error = f"Merge failed: {error}"
            return result

    result.merged_path = merged_path

    # Step 2: Convert to GGUF
    gguf_dir = config.paths.output.gguf / config.get_output_name()

    if not skip_gguf:
        success, gguf_path, error = convert_to_gguf(
            model_path=merged_path,
            output_path=gguf_dir,
            quantization=config.export.gguf.quantization,
            llama_cpp_path=config.export.gguf.llama_cpp_path,
        )
        if not success:
            result.error = f"GGUF conversion failed: {error}"
            return result

        result.gguf_path = gguf_path
    else:
        # Try to find existing GGUF
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            result.gguf_path = gguf_files[0]

    # Step 3: Create Ollama model
    if not skip_ollama and result.gguf_path:
        ollama_name = config.get_ollama_model_name()

        # Get chat template from config (default to auto-detection)
        chat_template = getattr(config.training, 'chat_template', 'auto') or 'auto'

        success, error = create_ollama_model(
            gguf_path=result.gguf_path,
            model_name=ollama_name,
            temperature=config.export.ollama.temperature,
            repeat_penalty=config.export.ollama.repeat_penalty,
            num_predict=config.export.ollama.num_predict,
            system_prompt=config.export.ollama.system_prompt,
            base_model=config.training.base_model,
            chat_template=chat_template,
        )
        if not success:
            result.error = f"Ollama import failed: {error}"
            return result

        result.ollama_model = ollama_name

    result.success = True
    return result


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.config_loader import load_config

    parser = argparse.ArgumentParser(description="Export trained model to Ollama")
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Path to pipeline config")
    parser.add_argument("--checkpoint", help="Path to checkpoint (auto-detected if not specified)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama import")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    result = export_model(
        config=config,
        checkpoint_path=checkpoint,
        skip_merge=args.skip_merge,
        skip_gguf=args.skip_gguf,
        skip_ollama=args.skip_ollama,
    )

    if result.success:
        print("\n" + "=" * 50)
        print("Export Complete!")
        print("=" * 50)
        if result.merged_path:
            print(f"Merged model: {result.merged_path}")
        if result.gguf_path:
            print(f"GGUF file: {result.gguf_path}")
        if result.ollama_model:
            print(f"Ollama model: {result.ollama_model}")
            print(f"\nRun with: ollama run {result.ollama_model}")
    else:
        print(f"\nExport failed: {result.error}")
        sys.exit(1)
