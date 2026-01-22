#!/usr/bin/env python3
"""
Model Evaluation Script

Tests the fine-tuned model against the base model and generates
a comprehensive markdown report with before/after comparisons.

Usage:
    python scripts/evaluate_model.py --config config/pipeline_config.yaml
    python scripts/evaluate_model.py --model python-expert-v5 --base deepseek-r1:1.5b

Output:
    docs/<model_name>_training_results.md
"""

import argparse
import json
import os
import subprocess
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Comprehensive test prompts covering training data categories
TEST_PROMPTS = [
    # Kubernetes
    {
        "category": "Kubernetes",
        "prompt": "How do I create a Kubernetes deployment with 3 replicas using the Python client?",
        "expected_keywords": ["kubernetes", "client", "deployment", "replicas", "create"],
        "tags": ["kubernetes", "python", "deployment"],
    },
    {
        "category": "Kubernetes",
        "prompt": "Write Python code to watch for pod events in a specific namespace",
        "expected_keywords": ["watch", "pod", "namespace", "events", "kubernetes"],
        "tags": ["kubernetes", "watch", "events"],
    },
    # Docker
    {
        "category": "Docker",
        "prompt": "How do I build and push a Docker image using docker-py?",
        "expected_keywords": ["docker", "build", "push", "image", "client"],
        "tags": ["docker", "python", "build"],
    },
    # Terraform/IaC
    {
        "category": "Terraform/IaC",
        "prompt": "Show me a Terraform module for creating an AWS EKS cluster with managed node groups",
        "expected_keywords": ["module", "eks", "cluster", "node_group", "aws"],
        "tags": ["terraform", "aws", "eks"],
    },
    {
        "category": "Crossplane",
        "prompt": "How do I use Crossplane compositions to provision AWS RDS instances?",
        "expected_keywords": ["crossplane", "composition", "rds", "aws", "claim"],
        "tags": ["crossplane", "aws", "rds"],
    },
    # AWS
    {
        "category": "AWS",
        "prompt": "Write Python boto3 code to assume a role and list S3 buckets in another account",
        "expected_keywords": ["boto3", "assume_role", "sts", "s3", "list_buckets"],
        "tags": ["aws", "boto3", "iam", "cross-account"],
    },
    {
        "category": "AWS Security",
        "prompt": "How do I implement AWS KMS key policies with least privilege access?",
        "expected_keywords": ["kms", "key", "policy", "principal", "action"],
        "tags": ["aws", "kms", "security", "iam"],
    },
    # Python Best Practices
    {
        "category": "Python",
        "prompt": "Show me how to properly handle exceptions and implement retry logic in Python",
        "expected_keywords": ["try", "except", "retry", "exception", "raise"],
        "tags": ["python", "exceptions", "retry"],
    },
    {
        "category": "Python Async",
        "prompt": "Write a Python async function that fetches data from multiple APIs concurrently",
        "expected_keywords": ["async", "await", "asyncio", "gather", "aiohttp"],
        "tags": ["python", "asyncio", "concurrent"],
    },
    # Testing
    {
        "category": "Testing",
        "prompt": "How do I write pytest fixtures for testing a FastAPI application with a database?",
        "expected_keywords": ["pytest", "fixture", "fastapi", "client", "database"],
        "tags": ["pytest", "fastapi", "fixtures", "testing"],
    },
    # CI/CD
    {
        "category": "CI/CD",
        "prompt": "Create a GitHub Actions workflow that deploys to EKS using OIDC authentication",
        "expected_keywords": ["github", "actions", "oidc", "eks", "deploy"],
        "tags": ["github-actions", "eks", "oidc", "ci-cd"],
    },
    # GitOps
    {
        "category": "GitOps",
        "prompt": "Show me an ArgoCD ApplicationSet that generates apps from a Git repository",
        "expected_keywords": ["applicationset", "argocd", "git", "generator", "template"],
        "tags": ["argocd", "gitops", "applicationset"],
    },
    # Security
    {
        "category": "Network Security",
        "prompt": "Write a Kubernetes NetworkPolicy that allows only specific pod-to-pod communication",
        "expected_keywords": ["networkpolicy", "ingress", "egress", "podSelector", "namespaceSelector"],
        "tags": ["kubernetes", "network-policy", "security"],
    },
    # Observability
    {
        "category": "Observability",
        "prompt": "How do I set up Prometheus and Grafana on Kubernetes using the kube-prometheus-stack Helm chart?",
        "expected_keywords": ["prometheus", "grafana", "helm", "kube-prometheus", "values"],
        "tags": ["prometheus", "grafana", "kubernetes", "observability"],
    },
    # Platform Engineering
    {
        "category": "Platform Engineering",
        "prompt": "How do I create a Backstage software template for provisioning Kubernetes namespaces?",
        "expected_keywords": ["backstage", "template", "scaffolder", "parameters", "steps"],
        "tags": ["backstage", "platform", "templates"],
    },
]


def run_ollama(model: str, prompt: str, timeout: int = 120) -> Tuple[Optional[str], float]:
    """Run Ollama model and get response with timing."""
    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            return f"Error: {result.stderr}", elapsed

        return result.stdout.strip(), elapsed

    except subprocess.TimeoutExpired:
        return None, timeout
    except FileNotFoundError:
        return "Error: Ollama not found", 0
    except Exception as e:
        return f"Error: {str(e)}", time.time() - start_time


def check_model_exists(model: str) -> bool:
    """Check if a model exists in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        return model in result.stdout
    except Exception:
        return False


def calculate_keyword_coverage(response: str, keywords: list) -> float:
    """Calculate what percentage of expected keywords are present."""
    if not response:
        return 0.0

    response_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    return found / len(keywords) * 100


def check_repetition(response: str) -> float:
    """Check for repetitive patterns in response."""
    if not response or len(response) < 100:
        return 0.0

    sentences = re.split(r'[.!?]', response)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2:
        return 0.0

    unique_sentences = set(sentences)
    repetition_ratio = 1 - len(unique_sentences) / len(sentences)
    return repetition_ratio * 100


def evaluate_response(response: str, keywords: List[str]) -> Dict:
    """Evaluate response quality based on various metrics."""
    if response is None:
        return {
            "length": 0,
            "has_code": False,
            "has_explanation": False,
            "keyword_coverage": 0,
            "repetition": 100,
            "error": True,
            "score": 0,
        }

    is_error = response.startswith("Error:")

    metrics = {
        "length": len(response),
        "has_code": "```" in response or "def " in response or "import " in response,
        "has_explanation": len(response) > 200 and not is_error,
        "keyword_coverage": calculate_keyword_coverage(response, keywords),
        "repetition": check_repetition(response),
        "error": is_error,
    }

    # Score calculation (0-100)
    score = 0
    if not metrics["error"]:
        # Keyword coverage (up to 40 points)
        score += min(40, metrics["keyword_coverage"] * 0.4)
        # Has code (30 points)
        if metrics["has_code"]:
            score += 30
        # Has explanation (20 points)
        if metrics["has_explanation"]:
            score += 20
        # Low repetition bonus (10 points)
        if metrics["repetition"] < 20:
            score += 10

    metrics["score"] = round(score)
    return metrics


def run_evaluation(
    fine_tuned_model: str,
    base_model: str,
    prompts: List[Dict],
    timeout: int = 120,
) -> List[Dict]:
    """Run evaluation on both models."""
    results = []

    print(f"\nEvaluating {len(prompts)} prompts...")
    print(f"  Fine-tuned: {fine_tuned_model}")
    print(f"  Base model: {base_model}")
    print("-" * 60)

    for i, test in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {test['category']}: {test['prompt'][:50]}...")

        keywords = test.get("expected_keywords", test.get("tags", []))

        # Query fine-tuned model
        print("  Testing fine-tuned model...", end=" ", flush=True)
        ft_response, ft_time = run_ollama(fine_tuned_model, test["prompt"], timeout)
        ft_metrics = evaluate_response(ft_response, keywords)
        print(f"Score: {ft_metrics['score']}, Time: {ft_time:.1f}s")

        # Query base model
        print("  Testing base model...", end=" ", flush=True)
        base_response, base_time = run_ollama(base_model, test["prompt"], timeout)
        base_metrics = evaluate_response(base_response, keywords)
        print(f"Score: {base_metrics['score']}, Time: {base_time:.1f}s")

        results.append({
            "category": test["category"],
            "prompt": test["prompt"],
            "tags": test.get("tags", []),
            "keywords": keywords,
            "fine_tuned": {
                "response": ft_response or "TIMEOUT",
                "time": ft_time,
                "metrics": ft_metrics,
            },
            "base": {
                "response": base_response or "TIMEOUT",
                "time": base_time,
                "metrics": base_metrics,
            },
        })

    return results


def generate_markdown_report(
    results: List[Dict],
    fine_tuned_model: str,
    base_model: str,
    config: Dict,
    output_path: Path,
) -> None:
    """Generate markdown report with results."""

    # Calculate summary statistics
    ft_scores = [r["fine_tuned"]["metrics"]["score"] for r in results]
    base_scores = [r["base"]["metrics"]["score"] for r in results]
    ft_times = [r["fine_tuned"]["time"] for r in results]
    base_times = [r["base"]["time"] for r in results]

    avg_ft_score = sum(ft_scores) / len(ft_scores) if ft_scores else 0
    avg_base_score = sum(base_scores) / len(base_scores) if base_scores else 0
    avg_ft_time = sum(ft_times) / len(ft_times) if ft_times else 0
    avg_base_time = sum(base_times) / len(base_times) if base_times else 0

    improvement = avg_ft_score - avg_base_score
    improvement_pct = (improvement / avg_base_score * 100) if avg_base_score > 0 else 0

    # Group by category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"ft_scores": [], "base_scores": []}
        categories[cat]["ft_scores"].append(r["fine_tuned"]["metrics"]["score"])
        categories[cat]["base_scores"].append(r["base"]["metrics"]["score"])

    # Count tests with code
    ft_with_code = sum(1 for r in results if r['fine_tuned']['metrics']['has_code'])
    base_with_code = sum(1 for r in results if r['base']['metrics']['has_code'])

    # Get config stats
    so_tags = len(config.get('external_sources', {}).get('stackoverflow', {}).get('tags', []))
    gh_repos = len(config.get('external_sources', {}).get('github', {}).get('repos', []))
    gh_searches = len(config.get('external_sources', {}).get('github', {}).get('searches', []))

    # Generate report
    report = f"""# Training Results: {fine_tuned_model}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | Fine-tuned | Base Model | Improvement |
|--------|------------|------------|-------------|
| **Average Score** | {avg_ft_score:.1f}/100 | {avg_base_score:.1f}/100 | {improvement:+.1f} ({improvement_pct:+.1f}%) |
| **Avg Response Time** | {avg_ft_time:.1f}s | {avg_base_time:.1f}s | {avg_ft_time - avg_base_time:+.1f}s |
| **Tests with Code** | {ft_with_code}/{len(results)} | {base_with_code}/{len(results)} | {ft_with_code - base_with_code:+d} |
| **Tests Improved** | {sum(1 for r in results if r['fine_tuned']['metrics']['score'] > r['base']['metrics']['score'])}/{len(results)} | - | - |
| **Tests Regressed** | {sum(1 for r in results if r['fine_tuned']['metrics']['score'] < r['base']['metrics']['score'])}/{len(results)} | - | - |

## Models Compared

| | Model |
|--|-------|
| **Fine-tuned** | `{fine_tuned_model}` |
| **Base** | `{base_model}` |

## Training Data Sources

| Source | Count |
|--------|-------|
| Stack Overflow Tags | {so_tags} |
| GitHub Static Repos | {gh_repos} |
| GitHub Search Queries | {gh_searches} |

## Results by Category

| Category | Fine-tuned Avg | Base Model Avg | Improvement |
|----------|----------------|----------------|-------------|
"""

    for cat, scores in sorted(categories.items()):
        ft_avg = sum(scores["ft_scores"]) / len(scores["ft_scores"])
        base_avg = sum(scores["base_scores"]) / len(scores["base_scores"])
        imp = ft_avg - base_avg
        emoji = "✅" if imp > 5 else "❌" if imp < -5 else "➖"
        report += f"| {cat} | {ft_avg:.1f} | {base_avg:.1f} | {emoji} {imp:+.1f} |\n"

    report += """
---

## Before/After Examples

"""

    # Add 3 best improved examples
    sorted_by_improvement = sorted(
        results,
        key=lambda r: r["fine_tuned"]["metrics"]["score"] - r["base"]["metrics"]["score"],
        reverse=True
    )

    report += "### Most Improved Examples\n\n"

    for r in sorted_by_improvement[:3]:
        ft_score = r["fine_tuned"]["metrics"]["score"]
        base_score = r["base"]["metrics"]["score"]
        improvement = ft_score - base_score

        report += f"""#### {r['category']}: {r['prompt'][:60]}...

**Improvement:** {base_score} → {ft_score} ({improvement:+d} points)

<details>
<summary>📗 Fine-tuned Response (Score: {ft_score})</summary>

```
{r['fine_tuned']['response'][:1500]}{'...[truncated]' if len(r['fine_tuned']['response']) > 1500 else ''}
```

</details>

<details>
<summary>📕 Base Model Response (Score: {base_score})</summary>

```
{r['base']['response'][:1500]}{'...[truncated]' if len(r['base']['response']) > 1500 else ''}
```

</details>

---

"""

    # Add detailed results table
    report += """## Detailed Test Results

| # | Category | Prompt | Fine-tuned | Base | Diff | Status |
|---|----------|--------|------------|------|------|--------|
"""

    for i, r in enumerate(results, 1):
        ft_score = r["fine_tuned"]["metrics"]["score"]
        base_score = r["base"]["metrics"]["score"]
        diff = ft_score - base_score

        if diff > 10:
            status = "✅ Improved"
        elif diff < -10:
            status = "❌ Regressed"
        else:
            status = "➖ Similar"

        prompt_short = r["prompt"][:40] + "..." if len(r["prompt"]) > 40 else r["prompt"]
        report += f"| {i} | {r['category']} | {prompt_short} | {ft_score} | {base_score} | {diff:+d} | {status} |\n"

    # Add conclusion
    if improvement_pct > 20:
        verdict = "🎉 **SIGNIFICANT IMPROVEMENT** - The fine-tuned model substantially outperforms the base model."
        recommendation = "The model is ready for production use."
    elif improvement_pct > 5:
        verdict = "✅ **MODERATE IMPROVEMENT** - The fine-tuned model shows meaningful gains."
        recommendation = "Consider additional training data to further improve performance."
    elif improvement_pct > -5:
        verdict = "➖ **SIMILAR PERFORMANCE** - The models perform comparably."
        recommendation = "Review training data quality and consider adding more diverse examples."
    else:
        verdict = "⚠️ **REGRESSION DETECTED** - The fine-tuned model underperforms the base model."
        recommendation = "Investigate training data for quality issues or consider reducing epochs."

    report += f"""
---

## Conclusion

{verdict}

### Recommendations

{recommendation}

### Key Statistics

- **Total Tests:** {len(results)}
- **Average Improvement:** {improvement:+.1f} points ({improvement_pct:+.1f}%)
- **Best Category:** {max(categories.items(), key=lambda x: sum(x[1]['ft_scores'])/len(x[1]['ft_scores']))[0]}
- **Most Improved:** {max(categories.items(), key=lambda x: sum(x[1]['ft_scores'])/len(x[1]['ft_scores']) - sum(x[1]['base_scores'])/len(x[1]['base_scores']))[0]}

---

*Generated by `scripts/evaluate_model.py` on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n📄 Report saved to: {output_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        print("Warning: PyYAML not available, using empty config")
        return {}

    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model and generate report")
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Pipeline config file")
    parser.add_argument("--model", help="Fine-tuned model name (overrides config)")
    parser.add_argument("--base", help="Base model name (overrides config)")
    parser.add_argument("--output", help="Output markdown file path")
    parser.add_argument("--limit", type=int, help="Limit number of tests")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per test (seconds)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine model names from config or args
    pipeline_config = config.get("pipeline", {})
    export_config = config.get("export", {}).get("ollama", {})
    training_config = config.get("training", {})

    model_name = export_config.get("model_name", "python-expert")
    version = pipeline_config.get("version", "v1")

    fine_tuned_model = args.model or f"{model_name}-{version}"

    # Map base model to Ollama name
    base_model_full = training_config.get("base_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    base_model_map = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-r1:1.5b",
        "Qwen/Qwen2.5-1.5B": "qwen2.5:1.5b",
        "meta-llama/Llama-3.2-1B": "llama3.2:1b",
    }
    base_model = args.base or base_model_map.get(base_model_full, "deepseek-r1:1.5b")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        safe_name = fine_tuned_model.replace(":", "-").replace("/", "-")
        output_path = Path(f"docs/{safe_name}_training_results.md")

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Fine-tuned model: {fine_tuned_model}")
    print(f"Base model: {base_model}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Check models exist
    if not check_model_exists(fine_tuned_model):
        print(f"\n⚠️  Fine-tuned model '{fine_tuned_model}' not found in Ollama")
        print("\nAvailable models:")
        subprocess.run(["ollama", "list"])
        return 1

    if not check_model_exists(base_model):
        print(f"\n📥 Base model '{base_model}' not found. Pulling...")
        subprocess.run(["ollama", "pull", base_model])

    # Run evaluation
    prompts = TEST_PROMPTS[:args.limit] if args.limit else TEST_PROMPTS
    results = run_evaluation(fine_tuned_model, base_model, prompts, args.timeout)

    # Generate markdown report
    generate_markdown_report(results, fine_tuned_model, base_model, config, output_path)

    # Print summary
    ft_avg = sum(r["fine_tuned"]["metrics"]["score"] for r in results) / len(results)
    base_avg = sum(r["base"]["metrics"]["score"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Fine-tuned: {ft_avg:.1f}/100")
    print(f"Base model: {base_avg:.1f}/100")
    print(f"Improvement: {ft_avg - base_avg:+.1f} points")
    print(f"{'=' * 60}")
    print(f"\n📄 Full report: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
