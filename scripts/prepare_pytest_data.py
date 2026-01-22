#!/usr/bin/env python3
"""
Convert pytest documentation to Alpaca training format.

Extracts:
- Code examples from RST files
- Python example files
- Q&A pairs from documentation
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def extract_code_blocks(content: str) -> List[Tuple[str, str]]:
    """Extract code blocks from RST content with their context."""
    blocks = []

    # Match .. code-block:: python or .. code:: python
    pattern = r'(?:^|\n)(.*?)\n\n\.\.\s+(?:code-block|code)::\s*python\s*\n((?:\s+.*\n?)+)'

    for match in re.finditer(pattern, content, re.MULTILINE):
        context = match.group(1).strip()
        code = match.group(2)

        # Dedent the code block
        lines = code.split('\n')
        if lines:
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

        code = code.strip()
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

        lines = code.split('\n')
        if lines:
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

        code = code.strip()
        if code and len(code) > 20:
            blocks.append((context, code))

    return blocks


def extract_title(content: str) -> str:
    """Extract title from RST file."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip() and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line and all(c in '=-~^' for c in next_line.strip()):
                return line.strip()
    return ""


def generate_pytest_instructions(context: str, title: str) -> List[str]:
    """Generate instruction variations for pytest examples."""
    instructions = []

    # Clean up context
    context_clean = context.replace('\n', ' ').strip()
    context_clean = re.sub(r'\s+', ' ', context_clean)

    if context_clean:
        instructions.append(f"Write pytest code to {context_clean.lower()}")
        instructions.append(f"Show me how to {context_clean.lower()} in pytest")

    if title:
        title_clean = title.lower().replace('_', ' ')
        instructions.append(f"Give me an example of {title_clean} with pytest")

    # Generic pytest instructions
    instructions.extend([
        "Write a pytest test example",
        "Show me pytest best practices",
    ])

    return instructions[:2]  # Limit to 2 variations


def process_rst_file(filepath: Path) -> List[Dict]:
    """Process a single RST file and generate training examples."""
    examples = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    title = extract_title(content)
    code_blocks = extract_code_blocks(content)

    for context, code in code_blocks:
        if len(code) < 30:  # Skip very short examples
            continue

        instructions = generate_pytest_instructions(context, title)

        for instruction in instructions:
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": code
            })

    # Also create Q&A style examples from the title
    if title and code_blocks:
        _, first_code = code_blocks[0]
        examples.append({
            "instruction": f"How do I implement {title.lower()} in pytest?",
            "input": "",
            "output": f"Here's an example of {title.lower()}:\n\n```python\n{first_code}\n```"
        })

    return examples


def process_python_file(filepath: Path) -> List[Dict]:
    """Process a Python example file."""
    examples = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    # Skip very short files
    if len(content) < 50:
        return []

    # Extract module docstring
    docstring = ""
    if content.startswith('"""'):
        end = content.find('"""', 3)
        if end > 0:
            docstring = content[3:end].strip()
    elif content.startswith("'''"):
        end = content.find("'''", 3)
        if end > 0:
            docstring = content[3:end].strip()

    filename = filepath.stem.replace('_', ' ')

    instructions = []
    if docstring:
        instructions.append(f"Write pytest code that {docstring.lower()}")
    instructions.append(f"Show me a pytest example for {filename}")
    instructions.append(f"Write a {filename} test with pytest")

    for instruction in instructions[:2]:
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": content
        })

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert pytest docs to Alpaca format")
    parser.add_argument("--input-dir", required=True, help="Path to pytest doc/en directory")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    all_examples = []

    # Process RST files
    rst_files = list(input_dir.rglob("*.rst"))
    print(f"Found {len(rst_files)} RST files")

    for filepath in rst_files:
        # Skip changelog and deprecations (too long/noisy)
        if filepath.name in ['changelog.rst', 'deprecations.rst']:
            continue

        examples = process_rst_file(filepath)
        if examples:
            print(f"  {filepath.name}: {len(examples)} examples")
            all_examples.extend(examples)

    # Process Python example files
    py_files = list(input_dir.rglob("*.py"))
    print(f"\nFound {len(py_files)} Python files")

    for filepath in py_files:
        if filepath.name in ['conf.py', 'conftest.py']:
            continue

        examples = process_python_file(filepath)
        if examples:
            print(f"  {filepath.name}: {len(examples)} examples")
            all_examples.extend(examples)

    print(f"\nTotal examples: {len(all_examples)}")

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
