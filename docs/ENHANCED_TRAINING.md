# Enhanced Training Strategy for DevOps Assistant

This document explains our improved approach to training data collection and model training for creating the ultimate DevOps assistant.

## Why Basic Training Data Isn't Enough

Our initial training approach had several limitations:

| Problem | Impact |
|---------|--------|
| **Code-only responses** | Model dumps code without explanation |
| **Single instruction style** | "Write Python code that..." is repetitive |
| **No negative examples** | Model doesn't learn what NOT to do |
| **Limited domains** | Mostly K8s Python client, missing broader DevOps |
| **No conversational training** | Model doesn't sound like a helpful engineer |

## The Enhanced Approach

### 1. Multi-Style Responses

Instead of one response per example, we generate multiple styles:

```python
# Original (boring, code-only):
{
  "instruction": "Write Python code to list pods",
  "output": "from kubernetes import client\nv1 = client.CoreV1Api()..."
}

# Enhanced - Conversational (like talking to a senior engineer):
{
  "instruction": "How do I list pods in Kubernetes?",
  "output": "Good question! Here's how I'd do it:\n\n```python\n...\n```\n\n**Key points:**\n- Always reuse the client connection..."
}

# Enhanced - Tutorial (step-by-step):
{
  "instruction": "Can you walk me through listing Kubernetes pods in Python?",
  "output": "Let me break this down step by step.\n\n**Step 1: Set up the client**\n..."
}

# Enhanced - Quick Reference (concise):
{
  "instruction": "K8s Python list pods",
  "output": "```python\nfrom kubernetes import client, config\nconfig.load_kube_config()\nv1.list_pod_for_all_namespaces()\n```"
}
```

### 2. DPO (Preference) Training

DPO teaches the model to prefer better responses. For each topic, we create:
- **Chosen**: The best practice approach with explanations
- **Rejected**: The common but problematic approach

Example:

```json
{
  "prompt": "How do I handle exceptions in Python?",
  "rejected": "Just catch everything:\n```python\ntry:\n    do_something()\nexcept:\n    pass\n```",
  "chosen": "Use specific exception handling:\n```python\ntry:\n    do_something()\nexcept FileNotFoundError:\n    logger.warning('File not found')\nexcept PermissionError as e:\n    logger.error(f'Permission denied: {e}')\n    raise\n```\n\n**Why:** Never use bare `except:` - it hides bugs..."
}
```

### 3. High-Quality Data Sources

| Source | Type | Value |
|--------|------|-------|
| Stack Overflow | Q&A | Real problems and solutions |
| GitHub Issues | Problem/Solution | Edge cases and fixes |
| Official Docs | Reference | Authoritative examples |
| Curated Templates | Patterns | Best practices with explanations |
| Anti-Patterns | Negative Examples | What NOT to do |

## Running the Enhanced Pipeline

### Step 1: Generate DPO Data

```bash
uv run python scripts/generate_dpo_data.py \
  --output data/training/dpo_pairs.json \
  --format dpo
```

### Step 2: Collect External Data

```bash
# Set API keys (optional but recommended for more data)
export STACKOVERFLOW_API_KEY="your_key"
export GITHUB_TOKEN="your_token"

uv run python scripts/collect_devops_data.py \
  --output data/training
```

### Step 3: Enhance Existing Data

```bash
uv run python scripts/enhanced_data_generator.py \
  --input data/training/all_python_deduped.json \
  --output data/training/all_python_enhanced.json \
  --domain python
```

### Step 4: Combine All Data

```bash
uv run python -c "
import json
from pathlib import Path

files = [
    'data/training/all_python_enhanced.json',
    'data/training/devops_training_data.json',
]

combined = []
for f in files:
    if Path(f).exists():
        with open(f) as fp:
            combined.extend(json.load(fp))

print(f'Total examples: {len(combined)}')

with open('data/training/combined_enhanced.json', 'w') as fp:
    json.dump(combined, fp, indent=2)
"
```

### Step 5: Train with DPO (Two-Stage)

**Stage 1: Supervised Fine-Tuning (SFT)**
```bash
uv run python -m pipeline.runner --config config/pipeline_config.yaml --stage train
```

**Stage 2: DPO Training** (after SFT)
```yaml
# In your axolotl config:
rl: dpo
datasets:
  - path: data/training/dpo_pairs.json
    type: preference
```

## Data Quality Guidelines

### Good Instruction Examples

| Bad | Good |
|-----|------|
| "Write Python code that lists pods" | "How do I list pods in Kubernetes?" |
| "Create a function called X" | "I need to check if a pod is running" |
| "Show me how to X" | "What's the best way to handle API errors?" |

### Good Response Examples

| Bad | Good |
|-----|------|
| Just code, no explanation | Code + why this approach |
| No error handling | Shows error handling patterns |
| Single approach | Compares alternatives |
| No context | Explains when to use this |

## Categories Covered

### Python Fundamentals
- Exception handling
- Context managers
- Type hints
- Async/await
- Testing patterns

### Kubernetes
- Client setup and reuse
- Watch vs polling
- Error handling
- RBAC and permissions
- Resource management

### Docker
- Dockerfile best practices
- Multi-stage builds
- Security (non-root)
- Image optimization

### Terraform
- State management
- Module patterns
- Variable handling
- Provider configuration

### Security
- No hardcoded secrets
- SQL injection prevention
- Input validation
- Authentication patterns

## Expected Results

With this enhanced approach, you should see:

| Metric | Before | After |
|--------|--------|-------|
| Response naturalness | Robotic | Conversational |
| Explanation quality | None | Detailed |
| Security awareness | Low | High |
| Best practice adherence | Random | Consistent |
| Error handling | Missing | Comprehensive |

## Tips for Further Improvement

1. **More data is better** - Aim for 10K+ high-quality examples
2. **Diverse styles** - Mix conversational, tutorial, and quick-ref
3. **Real problems** - Stack Overflow questions are gold
4. **Negative examples** - DPO pairs teach nuance
5. **Larger base model** - 7B+ will retain more learning

## File Structure

```
scripts/
├── enhanced_data_generator.py  # Multi-style response generator
├── collect_devops_data.py      # Stack Overflow + GitHub collector
└── generate_dpo_data.py        # DPO pair generator

data/training/
├── all_python_enhanced.json    # Enhanced basic examples
├── devops_training_data.json   # External source data
├── dpo_pairs.json              # Preference training pairs
└── combined_enhanced.json      # Final combined dataset
```

## Next Steps

1. Run the enhanced pipeline
2. Train with SFT + DPO
3. Evaluate on held-out DevOps questions
4. Iterate based on failure cases
