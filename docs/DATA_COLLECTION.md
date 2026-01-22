# Data Collection Pipeline

This document explains how to collect training data for fine-tuning LLMs on Python code.

## Overview

There are two approaches to collecting training data:

### 1. Modular Collectors (Recommended)

The `collectors/` module provides a pluggable architecture for collecting data from various sources:

- **Stack Overflow** - Q&A from Stack Exchange API
- **GitHub** - Issues, discussions, and code examples
- **Web Crawler** - Documentation sites with JavaScript support

### 2. Repository Extraction (Legacy)

The original pipeline for extracting data from cloned GitHub repositories:
1. **Clone** - Download repositories from GitHub
2. **Extract** - Parse Python files and documentation into training format
3. **Combine & Deduplicate** - Merge datasets and remove duplicates

---

## Modular Collectors

### Quick Start

```bash
# List available collectors
uv run python -m collectors --list

# Run a specific collector
uv run python -m collectors stackoverflow --output data/training/

# Run multiple collectors
uv run python -m collectors stackoverflow github --output data/training/

# Run all collectors
uv run python -m collectors --all --output data/training/
```

### Available Collectors

| Collector | Source | Description |
|-----------|--------|-------------|
| `stackoverflow` | Stack Exchange API | High-quality Q&A by tag (python, kubernetes, docker, etc.) |
| `github` | GitHub API | Issues with solutions, Discussions, code examples |
| `web_crawler` | Documentation sites | JavaScript-rendered pages via Playwright |

### Collector Configuration

Each collector supports custom configuration via JSON:

```bash
# Show config options for a collector
uv run python -m collectors stackoverflow --help-config

# Run with custom config
uv run python -m collectors stackoverflow --config '{"min_score": 20, "max_per_tag": 100}'

# Use a config file
uv run python -m collectors github --config config/github_config.json
```

### Environment Variables

Create a `.env` file in the project root (gitignored):

```bash
# Stack Exchange API key (optional, increases rate limits)
STACKOVERFLOW_API_KEY=your_key_here

# GitHub personal access token (required for discussions)
GITHUB_TOKEN=ghp_your_token_here
```

### Stack Overflow Collector

Collects high-quality Q&A from Stack Overflow by tag.

```bash
# Default tags: python, kubernetes, docker, terraform, ansible, aws, boto3, pytest, flask, fastapi
uv run python -m collectors stackoverflow -o data/training/

# Custom tags
uv run python -m collectors stackoverflow --config '{"tags": ["python", "asyncio", "fastapi"]}'
```

**Configuration:**
- `api_key`: Stack Exchange API key (optional)
- `tags`: List of tags to collect (default: DevOps-focused)
- `min_score`: Minimum question score (default: 10)
- `max_per_tag`: Maximum questions per tag (default: 200)
- `site`: Stack Exchange site (default: "stackoverflow")

### GitHub Collector

Collects from GitHub issues, discussions, and code examples. Supports **repository search** to automatically find top repos.

```bash
# Default repos: kubernetes-client/python, docker/docker-py, hashicorp/terraform, etc.
uv run python -m collectors github -o data/training/

# Search for repos by query (finds top 100 repos with 100+ stars)
uv run python -m collectors.github --search "python kubernetes" --search "terraform aws" -o data/training/

# Search only (no default repos)
uv run python -m collectors.github --search "python devops" --no-default-repos -o data/training/

# Custom search parameters
uv run python -m collectors.github --search "fastapi" --max-repos 50 --min-stars 500 -o data/training/

# Custom repos (no search)
uv run python -m collectors github --config '{"repos": ["fastapi/fastapi", "pallets/flask"]}'
```

**Configuration:**
- `token`: GitHub personal access token (from env or config)
- `repos`: List of repos to collect from
- `searches`: List of search queries to find repos (e.g., `["python devops", "terraform aws"]`)
- `max_repos_per_search`: Max repos per search query (default: 100)
- `min_stars`: Minimum stars for searched repos (default: 100)
- `max_issues_per_repo`: Maximum issues per repo (default: 50)
- `collect_discussions`: Whether to collect discussions (default: true)
- `collect_code`: Whether to collect code examples (default: false)

### Web Crawler Collector

Crawls documentation sites with full JavaScript support via Playwright.

```bash
# Install Playwright browsers first
playwright install chromium

# Crawl default documentation sites
uv run python -m collectors web_crawler -o data/training/

# Custom URLs
uv run python -m collectors web_crawler --config '{"urls": ["https://docs.python.org/3/tutorial/"]}'
```

**Configuration:**
- `urls`: List of starting URLs to crawl
- `max_pages`: Maximum pages per domain (default: 100)
- `max_depth`: Maximum crawl depth (default: 3)
- `js_wait`: Seconds to wait for JavaScript (default: 2)
- `follow_links`: Whether to follow links (default: true)
- `headless`: Run browser in headless mode (default: true)

### Adding a New Collector

1. Create `collectors/my_source.py`
2. Inherit from `BaseCollector`
3. Implement required methods
4. Register with `@register_collector` decorator

```python
from collectors import BaseCollector, QAPair, register_collector

@register_collector
class MySourceCollector(BaseCollector):
    def get_name(self) -> str:
        return "my_source"

    def get_description(self) -> str:
        return "Collects data from my source"

    def collect(self):
        # Yield raw data items
        for item in self.fetch_data():
            yield item

    def transform(self, raw_item) -> QAPair:
        # Convert to training format
        return QAPair(
            instruction="How do I...",
            input="",
            output="Here's how...",
            source="my_source:123",
            source_type="my_source"
        )
```

### Output Formats

All collectors support multiple output formats:

```bash
# Alpaca format (default) - for SFT training
uv run python -m collectors stackoverflow -o data/ --format alpaca

# ShareGPT format - for multi-turn conversations
uv run python -m collectors stackoverflow -o data/ --format sharegpt

# Full format - includes all metadata
uv run python -m collectors stackoverflow -o data/ --format full
```

---

## Repository Extraction (Legacy)

The legacy pipeline extracts training data from cloned GitHub repositories.

## Configuration

### Repository URLs

Edit the following file to configure which repositories to process:

**File: `config/data_sources.yaml`**

```yaml
# Python Code Repositories
repositories:
  # Kubernetes Python Client Examples
  - name: k8s-python
    url: https://github.com/kubernetes-client/python
    subdirs:
      - examples
    type: python

  # Pytest Documentation
  - name: pytest
    url: https://github.com/pytest-dev/pytest
    subdirs:
      - doc/en
    type: mixed  # Python + RST docs

  # Advanced Python Tutorials
  - name: advanced-python-krother
    url: https://github.com/krother/advanced_python
    type: python

  # Python Mastery Course
  - name: python-mastery
    url: https://github.com/dabeaz-course/python-mastery
    type: python

  # Amazing Python Scripts Collection
  - name: amazing-python
    url: https://github.com/avinashkranjan/Amazing-Python-Scripts
    type: python
```

## Directory Structure

```
llm_training_web_data/
├── config/
│   └── data_sources.yaml      # Configure URLs here
├── repos/                      # Cloned repositories
│   ├── k8s-python-client/
│   ├── pytest-repo/
│   ├── advanced_python/
│   ├── python_mastery/
│   └── amazing_python/
├── data/
│   ├── raw/                    # Raw extracted data
│   └── training/               # Processed training data
│       ├── k8s_python_alpaca.json
│       ├── pytest_alpaca.json
│       ├── advanced_python_1.json
│       ├── python_mastery.json
│       ├── amazing_python.json
│       ├── all_python_combined.json
│       └── all_python_deduped.json  # Final training dataset
└── scripts/
    ├── prepare_k8s_python_data.py   # Python file extractor
    ├── prepare_pytest_data.py       # RST/docs extractor
    └── deduplicate_dataset.py       # Deduplication script
```

## Step-by-Step Guide

### Step 1: Clone Repositories

```bash
# Create repos directory
mkdir -p repos

# Clone each repository (shallow clone for speed)
git clone --depth 1 https://github.com/kubernetes-client/python repos/k8s-python-client
git clone --depth 1 https://github.com/pytest-dev/pytest repos/pytest-repo
git clone --depth 1 https://github.com/krother/advanced_python repos/advanced_python
git clone --depth 1 https://github.com/dabeaz-course/python-mastery repos/python_mastery
git clone --depth 1 https://github.com/avinashkranjan/Amazing-Python-Scripts repos/amazing_python
```

### Step 2: Extract Training Data

#### For Python Files

```bash
# Extract from each repository
uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/k8s-python-client/examples" \
  --output "data/training/k8s_python_alpaca.json" \
  --include-subdirs

uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/advanced_python" \
  --output "data/training/advanced_python.json" \
  --include-subdirs

uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/python_mastery" \
  --output "data/training/python_mastery.json" \
  --include-subdirs

uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/amazing_python" \
  --output "data/training/amazing_python.json" \
  --include-subdirs
```

#### For Documentation (RST files)

```bash
uv run python scripts/prepare_pytest_data.py \
  --input-dir "repos/pytest-repo/doc/en" \
  --output "data/training/pytest_alpaca.json"
```

### Step 3: Combine Datasets

```bash
python3 -c "
import json
import os

datasets = [
    'data/training/k8s_python_alpaca.json',
    'data/training/pytest_alpaca.json',
    'data/training/advanced_python.json',
    'data/training/python_mastery.json',
    'data/training/amazing_python.json'
]

combined = []
for path in datasets:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            print(f'{os.path.basename(path)}: {len(data)} examples')
            combined.extend(data)

print(f'Total: {len(combined)}')

with open('data/training/all_python_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
"
```

### Step 4: Deduplicate

```bash
uv run python scripts/deduplicate_dataset.py \
  --input "data/training/all_python_combined.json" \
  --output "data/training/all_python_deduped.json" \
  --min-output-length 50
```

## How Data Extraction Works

### Python File Extraction (`prepare_k8s_python_data.py`)

The script uses Python's `ast` module to parse source files:

1. **Parse the file** using `ast.parse()`
2. **Extract module docstring** - describes what the file does
3. **Extract functions** - name, docstring, arguments, source code
4. **Generate instruction/output pairs**:

```python
# From docstring
{
    "instruction": "Write Python code that {docstring}",
    "input": "",
    "output": "<full file content>"
}

# From function
{
    "instruction": "Create a Python function called {func_name}",
    "input": "",
    "output": "<function source code>"
}

# Q&A style
{
    "instruction": "How do I {task} in Python?",
    "input": "",
    "output": "Here's how:\n```python\n{code}\n```"
}
```

### RST Documentation Extraction (`prepare_pytest_data.py`)

Parses reStructuredText documentation:

1. **Find code blocks** matching `.. code-block:: python`
2. **Extract surrounding context** as the instruction
3. **Generate training pairs** from code examples

## Deduplication Logic

The `deduplicate_dataset.py` script removes:

| Filter | Description |
|--------|-------------|
| Duplicate instructions | Same question asked multiple times |
| Duplicate outputs | Same code appearing multiple times |
| Repetitive outputs | Code with excessive repetition patterns |
| Short outputs | Outputs below minimum length threshold |

## Adding New Data Sources

### 1. Add a new Python repository

```bash
# Clone the repo
git clone --depth 1 https://github.com/user/repo repos/new_repo

# Extract training data
uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/new_repo" \
  --output "data/training/new_repo.json" \
  --include-subdirs

# Add to combination step
```

### 2. Add documentation source

If the repo contains RST/Markdown documentation with code examples:

```bash
uv run python scripts/prepare_pytest_data.py \
  --input-dir "repos/new_repo/docs" \
  --output "data/training/new_repo_docs.json"
```

### 3. Recombine and deduplicate

After adding new sources, re-run the combine and deduplicate steps.

## Output Format (Alpaca)

All training data is output in Alpaca format:

```json
[
  {
    "instruction": "Write a Python function to calculate factorial",
    "input": "",
    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
  },
  {
    "instruction": "Show me how to create a Kubernetes deployment in Python",
    "input": "",
    "output": "from kubernetes import client, config\n\nconfig.load_kube_config()\n..."
  }
]
```

## Current Data Statistics

| Source | Repository | Examples |
|--------|------------|----------|
| K8s Python | kubernetes-client/python | 182 |
| Pytest | pytest-dev/pytest | 713 |
| Advanced Python | krother/advanced_python | 220 |
| Python Mastery | dabeaz-course/python-mastery | 999 |
| Amazing Scripts | avinashkranjan/Amazing-Python-Scripts | 4,331 |
| **Total Raw** | - | **6,445** |
| **After Dedup** | - | **3,323** |

## Troubleshooting

### Syntax errors during extraction

Some Python files may have syntax errors. The scripts will skip these and report:
```
Syntax error in path/to/file.py: invalid syntax
```

### Empty extraction

If a repository returns 0 examples:
- Check if the path is correct
- Verify the repo contains `.py` files
- Some repos may only contain notebooks (`.ipynb`)

### Memory issues with large repos

For very large repositories, process subdirectories separately:
```bash
uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/large_repo/src" \
  --output "data/training/large_repo_src.json"

uv run python scripts/prepare_k8s_python_data.py \
  --input-dir "repos/large_repo/examples" \
  --output "data/training/large_repo_examples.json"
```
