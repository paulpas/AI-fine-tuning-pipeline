"""
Unified Data Extractor

Extracts training examples from various source types:
- Python files (functions, classes, full files)
- RST documentation (code blocks with context)
- Markdown files (code blocks)

Replaces: scripts/prepare_k8s_python_data.py, scripts/prepare_pytest_data.py

Usage:
    from pipeline.data_extractor import extract_from_source
    from pipeline.config_loader import load_config

    config = load_config()
    for source in config.get_enabled_git_sources():
        examples = extract_from_source(source, config)
"""

import ast
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extraction from a single source."""
    source_name: str
    examples: List[Dict[str, str]]
    files_processed: int
    errors: List[str]


# =============================================================================
# Python Extraction
# =============================================================================

def extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from an AST node."""
    return ast.get_docstring(node)


def extract_functions(source: str) -> List[Dict[str, Any]]:
    """Extract function definitions with their docstrings and source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    lines = source.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
            func_source = '\n'.join(lines[start_line:end_line])

            functions.append({
                'name': node.name,
                'docstring': ast.get_docstring(node),
                'source': func_source,
                'args': [arg.arg for arg in node.args.args],
                'is_private': node.name.startswith('_'),
            })

    return functions


def clean_code(code: str) -> str:
    """Remove license header and clean up code."""
    lines = code.split('\n')

    # Skip license header (consecutive comment lines at start)
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            continue
        else:
            start_idx = i
            break

    return '\n'.join(lines[start_idx:]).strip()


def generate_python_instructions(
    filename: str,
    docstring: Optional[str],
    context: str = ""
) -> List[str]:
    """Generate instruction variations for Python code."""
    base_name = Path(filename).stem.replace('_', ' ').replace('-', ' ')
    instructions = []

    if docstring:
        doc_clean = docstring.split('\n')[0].strip()  # First line only
        instructions.extend([
            f"Write Python code that {doc_clean.lower()}",
            f"Show me how to {doc_clean.lower()}",
            f"Create a Python script to {doc_clean.lower()}",
        ])

    instructions.extend([
        f"Write a Python example for {base_name}",
        f"Show me Python code for {base_name}",
    ])

    if context:
        instructions.append(f"Write Python code for {context}")

    return instructions[:3]  # Limit variations


def extract_from_python_file(
    filepath: Path,
    include_functions: bool = True,
    include_full_file: bool = True,
    min_length: int = 50,
    max_length: int = 10000,
) -> List[Dict[str, str]]:
    """Extract training examples from a single Python file."""
    examples = []

    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        log.warning(f"Error reading {filepath}: {e}")
        return []

    # Skip __init__.py or very short files
    if filepath.name == '__init__.py' or len(source.strip()) < min_length:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        log.debug(f"Syntax error in {filepath}: {e}")
        return []

    module_docstring = ast.get_docstring(tree)
    clean_source = clean_code(source)

    if len(clean_source) > max_length:
        clean_source = clean_source[:max_length] + "\n# ... (truncated)"

    # Full file examples
    if include_full_file and min_length <= len(clean_source) <= max_length:
        instructions = generate_python_instructions(filepath.name, module_docstring)
        for instruction in instructions[:2]:
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": clean_source,
            })

    # Function-level examples
    if include_functions:
        functions = extract_functions(source)
        for func in functions:
            if func['is_private'] or func['name'] == 'main':
                continue
            if len(func['source']) < min_length:
                continue

            func_instructions = generate_python_instructions(
                func['name'] + ".py",
                func['docstring'],
                context=func['name'].replace('_', ' ')
            )

            for instruction in func_instructions[:1]:
                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": func['source'],
                })

    return examples


# =============================================================================
# RST Extraction
# =============================================================================

def extract_rst_code_blocks(content: str) -> List[Tuple[str, str]]:
    """Extract code blocks from RST content with their context."""
    blocks = []

    # Match .. code-block:: python or .. code:: python
    pattern = r'(?:^|\n)(.*?)\n\n\.\.\s+(?:code-block|code)::\s*python\s*\n((?:\s+.*\n?)+)'

    for match in re.finditer(pattern, content, re.MULTILINE):
        context = match.group(1).strip()
        code = match.group(2)
        code = _dedent_code(code)

        if code and len(code) > 20:
            blocks.append((context, code))

    # Also match literal blocks (::)
    pattern2 = r'(?:^|\n)(.*?)::\s*\n\n((?:\s{4,}.*\n?)+)'
    for match in re.finditer(pattern2, content, re.MULTILINE):
        context = match.group(1).strip()
        code = match.group(2)

        # Check if it looks like Python code
        if not any(kw in code for kw in ['def ', 'import ', 'class ', 'pytest', 'assert']):
            continue

        code = _dedent_code(code)
        if code and len(code) > 20:
            blocks.append((context, code))

    return blocks


def _dedent_code(code: str) -> str:
    """Remove common indentation from code block."""
    lines = code.split('\n')
    if not lines:
        return ""

    # Find minimum indentation
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent < float('inf'):
        code = '\n'.join(
            line[min_indent:] if len(line) > min_indent else line.strip()
            for line in lines
        )

    return code.strip()


def extract_rst_title(content: str) -> str:
    """Extract title from RST file."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip() and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line and all(c in '=-~^' for c in next_line.strip()):
                return line.strip()
    return ""


def extract_from_rst_file(
    filepath: Path,
    min_length: int = 50,
) -> List[Dict[str, str]]:
    """Extract training examples from an RST file."""
    examples = []

    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        log.warning(f"Error reading {filepath}: {e}")
        return []

    title = extract_rst_title(content)
    code_blocks = extract_rst_code_blocks(content)

    for context, code in code_blocks:
        if len(code) < min_length:
            continue

        context_clean = re.sub(r'\s+', ' ', context).strip()

        instructions = []
        if context_clean:
            instructions.append(f"Write code to {context_clean.lower()}")
            instructions.append(f"Show me how to {context_clean.lower()}")
        if title:
            instructions.append(f"Give me an example of {title.lower()}")

        for instruction in instructions[:2]:
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": code,
            })

    return examples


# =============================================================================
# Markdown Extraction
# =============================================================================

def extract_markdown_code_blocks(content: str) -> List[Tuple[str, str]]:
    """Extract Python code blocks from Markdown content."""
    blocks = []

    # Match ```python ... ```
    pattern = r'(?:^|\n)([^\n]*)\n```(?:python|py)\n(.*?)```'

    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        context = match.group(1).strip()
        code = match.group(2).strip()

        if code and len(code) > 20:
            blocks.append((context, code))

    return blocks


def extract_from_markdown_file(
    filepath: Path,
    min_length: int = 50,
) -> List[Dict[str, str]]:
    """Extract training examples from a Markdown file."""
    examples = []

    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        log.warning(f"Error reading {filepath}: {e}")
        return []

    code_blocks = extract_markdown_code_blocks(content)

    for context, code in code_blocks:
        if len(code) < min_length:
            continue

        context_clean = re.sub(r'[#*`]', '', context).strip()

        instructions = []
        if context_clean:
            instructions.append(f"Write Python code that {context_clean.lower()}")
        instructions.append(f"Show me this Python example: {filepath.stem}")

        for instruction in instructions[:1]:
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": code,
            })

    return examples


# =============================================================================
# Unified Extraction Interface
# =============================================================================

def extract_from_directory(
    directory: Path,
    source_type: str = "python",
    include_subdirs: bool = True,
    min_length: int = 50,
    max_length: int = 10000,
    include_functions: bool = True,
    include_full_files: bool = True,
) -> ExtractionResult:
    """
    Extract training examples from a directory.

    Args:
        directory: Path to directory
        source_type: Type of source (python, rst, markdown, mixed)
        include_subdirs: Whether to recursively process subdirectories
        min_length: Minimum code length to include
        max_length: Maximum code length to include
        include_functions: Extract function-level examples
        include_full_files: Extract full file examples

    Returns:
        ExtractionResult with all examples
    """
    examples = []
    errors = []
    files_processed = 0

    if not directory.exists():
        return ExtractionResult(
            source_name=directory.name,
            examples=[],
            files_processed=0,
            errors=[f"Directory not found: {directory}"]
        )

    # Determine file patterns based on source type
    patterns = {
        "python": ["*.py"],
        "rst": ["*.rst"],
        "markdown": ["*.md", "*.markdown"],
        "mixed": ["*.py", "*.rst", "*.md"],
    }

    file_patterns = patterns.get(source_type, ["*.py"])

    # Collect files
    all_files = []
    for pattern in file_patterns:
        if include_subdirs:
            all_files.extend(directory.rglob(pattern))
        else:
            all_files.extend(directory.glob(pattern))

    log.info(f"Found {len(all_files)} files in {directory}")

    for filepath in all_files:
        # Skip common non-content files
        if filepath.name in ['__init__.py', 'conftest.py', 'conf.py', 'setup.py']:
            continue
        if 'test' in filepath.name.lower() and source_type != "python":
            continue

        try:
            if filepath.suffix == '.py':
                file_examples = extract_from_python_file(
                    filepath,
                    include_functions=include_functions,
                    include_full_file=include_full_files,
                    min_length=min_length,
                    max_length=max_length,
                )
            elif filepath.suffix == '.rst':
                file_examples = extract_from_rst_file(filepath, min_length=min_length)
            elif filepath.suffix in ['.md', '.markdown']:
                file_examples = extract_from_markdown_file(filepath, min_length=min_length)
            else:
                continue

            examples.extend(file_examples)
            files_processed += 1

            if file_examples:
                log.debug(f"  {filepath.name}: {len(file_examples)} examples")

        except Exception as e:
            errors.append(f"{filepath}: {e}")

    return ExtractionResult(
        source_name=directory.name,
        examples=examples,
        files_processed=files_processed,
        errors=errors,
    )


def extract_from_git_source(
    source: Any,  # GitSource from config
    repos_dir: Path,
    min_length: int = 50,
    max_length: int = 10000,
) -> ExtractionResult:
    """
    Extract training examples from a git source defined in config.

    Args:
        source: GitSource configuration object
        repos_dir: Base directory where repos are cloned
        min_length: Minimum code length
        max_length: Maximum code length

    Returns:
        ExtractionResult with all examples
    """
    repo_dir = repos_dir / source.name

    if not repo_dir.exists():
        return ExtractionResult(
            source_name=source.name,
            examples=[],
            files_processed=0,
            errors=[f"Repository not cloned: {repo_dir}"]
        )

    all_examples = []
    total_files = 0
    all_errors = []

    # Process each subdirectory (or entire repo if no subdirs specified)
    subdirs = source.subdirs if source.subdirs else [""]

    for subdir in subdirs:
        target_dir = repo_dir / subdir if subdir else repo_dir

        result = extract_from_directory(
            directory=target_dir,
            source_type=source.type,
            include_subdirs=True,
            min_length=min_length,
            max_length=max_length,
        )

        all_examples.extend(result.examples)
        total_files += result.files_processed
        all_errors.extend(result.errors)

    log.info(f"Extracted {len(all_examples)} examples from {source.name} ({total_files} files)")

    return ExtractionResult(
        source_name=source.name,
        examples=all_examples,
        files_processed=total_files,
        errors=all_errors,
    )


def save_examples(
    examples: List[Dict[str, str]],
    output_path: Path,
    format: str = "alpaca"
) -> None:
    """
    Save examples to a JSON file.

    Args:
        examples: List of training examples
        output_path: Path to output file
        format: Output format (alpaca, sharegpt)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "sharegpt":
        # Convert to ShareGPT format
        converted = []
        for ex in examples:
            converted.append({
                "conversations": [
                    {"from": "human", "value": ex["instruction"]},
                    {"from": "gpt", "value": ex["output"]},
                ]
            })
        examples = converted

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    log.info(f"Saved {len(examples)} examples to {output_path}")
