#!/usr/bin/env python3
"""
Convert Kubernetes Python client examples to Alpaca training format.

Generates multiple training examples per file:
- Full file examples with docstring-based instructions
- Function-level examples
- Task-specific Q&A pairs
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Optional


def extract_docstring(node) -> Optional[str]:
    """Extract docstring from an AST node."""
    if (node.body and isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value.strip()
    return None


def extract_functions(source: str) -> list:
    """Extract function definitions with their docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function source
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
            lines = source.split('\n')
            func_source = '\n'.join(lines[start_line:end_line])

            # Get docstring
            docstring = ast.get_docstring(node)

            functions.append({
                'name': node.name,
                'docstring': docstring,
                'source': func_source,
                'args': [arg.arg for arg in node.args.args]
            })

    return functions


def clean_code(code: str) -> str:
    """Remove license header and clean up code."""
    lines = code.split('\n')

    # Skip license header (lines starting with # until first non-comment)
    start_idx = 0
    in_license = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_license:
            if stripped.startswith('#') or stripped == '':
                continue
            else:
                start_idx = i
                break

    return '\n'.join(lines[start_idx:]).strip()


def generate_instructions(filename: str, docstring: str) -> list:
    """Generate various instruction phrasings for the same content."""
    base_name = Path(filename).stem.replace('_', ' ').replace('-', ' ')

    instructions = []

    # Based on docstring
    if docstring:
        doc_clean = docstring.replace('\n', ' ').strip()
        instructions.extend([
            f"Write Python code that {doc_clean.lower()}",
            f"Show me how to {doc_clean.lower()} using the Kubernetes Python client",
            f"Create a Python script to {doc_clean.lower()}",
        ])

    # Based on filename
    instructions.extend([
        f"Write a Python example for Kubernetes {base_name}",
        f"Show me Python code for {base_name} in Kubernetes",
    ])

    return instructions


def generate_function_instructions(func_name: str, docstring: str, args: list) -> list:
    """Generate instructions for function-level examples."""
    name_clean = func_name.replace('_', ' ')

    instructions = []

    if docstring:
        instructions.append(f"Write a Python function to {docstring.lower()}")

    instructions.extend([
        f"Create a Python function called {func_name} for Kubernetes",
        f"Show me how to implement {name_clean} in Python for Kubernetes",
    ])

    return instructions


def process_file(filepath: Path) -> list:
    """Process a single Python file and generate training examples."""
    examples = []

    try:
        source = filepath.read_text()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Parse the file
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []

    # Get module docstring
    module_docstring = ast.get_docstring(tree)

    # Clean the code (remove license header)
    clean_source = clean_code(source)

    # Skip __init__.py or empty files
    if filepath.name == '__init__.py' or len(clean_source) < 50:
        return []

    # Generate full-file examples
    instructions = generate_instructions(filepath.name, module_docstring)
    for instruction in instructions[:2]:  # Limit to 2 per file
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": clean_source
        })

    # Generate function-level examples
    functions = extract_functions(source)
    for func in functions:
        if func['name'].startswith('_') or func['name'] == 'main':
            continue  # Skip private functions and main

        func_instructions = generate_function_instructions(
            func['name'], func['docstring'], func['args']
        )

        for instruction in func_instructions[:1]:  # 1 per function
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": func['source']
            })

    # Generate Q&A style examples
    if module_docstring:
        qa_pairs = [
            (f"How do I {module_docstring.lower()} with Python?",
             f"Here's how to {module_docstring.lower()} using the Kubernetes Python client:\n\n```python\n{clean_source}\n```"),
            (f"What's the Python code for {Path(filepath).stem.replace('_', ' ')}?",
             clean_source),
        ]
        for q, a in qa_pairs:
            examples.append({
                "instruction": q,
                "input": "",
                "output": a
            })

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert K8s Python examples to Alpaca format")
    parser.add_argument("--input-dir", required=True, help="Directory containing Python examples")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--include-subdirs", action="store_true", help="Include subdirectories")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    # Collect all Python files
    if args.include_subdirs:
        python_files = list(input_dir.rglob("*.py"))
    else:
        python_files = list(input_dir.glob("*.py"))

    print(f"Found {len(python_files)} Python files")

    # Process each file
    all_examples = []
    for filepath in python_files:
        print(f"Processing: {filepath.name}")
        examples = process_file(filepath)
        all_examples.extend(examples)
        print(f"  Generated {len(examples)} examples")

    print(f"\nTotal examples: {len(all_examples)}")

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
