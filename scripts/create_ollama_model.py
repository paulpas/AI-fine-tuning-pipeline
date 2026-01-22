#!/usr/bin/env python3
"""
Import model into Ollama.
"""

import subprocess
import argparse
from pathlib import Path


def create_modelfile(
    gguf_path: str,
    output_dir: str,
    model_name: str,
    temperature: float = 0.5,
    repeat_penalty: float = 1.3,
    num_predict: int = 512,
    system_prompt: str = ""
):
    """Create Ollama Modelfile."""

    # Make path absolute
    gguf_path = str(Path(gguf_path).absolute())

    modelfile_content = f'''FROM {gguf_path}

TEMPLATE """{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}<|end|>
"""

PARAMETER temperature {temperature}
PARAMETER num_predict {num_predict}
PARAMETER repeat_penalty {repeat_penalty}
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
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error importing to Ollama: {result.stderr}")
        return False

    print(f"Model imported to Ollama: {model_name}")
    print(f"Run with: ollama run {model_name}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import model into Ollama")
    parser.add_argument("--gguf-path", required=True, help="Path to GGUF file")
    parser.add_argument("--output-dir", required=True, help="Output directory for Modelfile")
    parser.add_argument("--model-name", required=True, help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--repeat-penalty", type=float, default=1.3, help="Repeat penalty")
    parser.add_argument("--num-predict", type=int, default=512, help="Max tokens")
    parser.add_argument("--system-prompt", default="", help="System prompt")

    args = parser.parse_args()

    modelfile_path = create_modelfile(
        args.gguf_path,
        args.output_dir,
        args.model_name,
        args.temperature,
        args.repeat_penalty,
        args.num_predict,
        args.system_prompt
    )

    import_to_ollama(modelfile_path, args.model_name)
