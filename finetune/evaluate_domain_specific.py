#!/usr/bin/env python3
"""
Domain-specific evaluation focusing on HashiCorp Terraform knowledge
that should have been improved by fine-tuning on official documentation.
"""

import subprocess
import re
import time
import json
from pathlib import Path

# Domain-specific questions that test knowledge from HashiCorp docs
DOMAIN_QUESTIONS = [
    {
        "id": "moved_block",
        "question": "What is the 'moved' block in Terraform and when should I use it?",
        "expected_terms": ["moved", "from", "to", "refactor", "state", "rename"],
        "category": "advanced",
        "notes": "moved block is a newer Terraform feature for refactoring without state manipulation",
    },
    {
        "id": "import_block",
        "question": "How do I use the 'import' block in Terraform to import existing resources?",
        "expected_terms": ["import", "id", "to", "resource", "existing"],
        "category": "advanced",
        "notes": "import block is a declarative way to import (Terraform 1.5+)",
    },
    {
        "id": "terraform_cloud",
        "question": "How do I configure Terraform Cloud as a remote backend?",
        "expected_terms": ["cloud", "organization", "workspaces", "hostname", "app.terraform.io"],
        "category": "cloud",
        "notes": "Tests knowledge of Terraform Cloud configuration",
    },
    {
        "id": "check_block",
        "question": "What is the 'check' block in Terraform and how does it differ from preconditions?",
        "expected_terms": ["check", "assert", "condition", "error_message", "continuous"],
        "category": "advanced",
        "notes": "check block is for continuous validation (Terraform 1.5+)",
    },
    {
        "id": "provider_meta",
        "question": "What is provider_meta in Terraform modules and when would you use it?",
        "expected_terms": ["provider_meta", "terraform_data", "module", "provider"],
        "category": "advanced",
        "notes": "Provider meta-arguments for advanced module patterns",
    },
    {
        "id": "replace_triggered",
        "question": "Explain the replace_triggered_by lifecycle argument in Terraform.",
        "expected_terms": ["replace_triggered_by", "lifecycle", "replacement", "resource", "change"],
        "category": "lifecycle",
        "notes": "Newer lifecycle argument for triggering replacements",
    },
    {
        "id": "sensitive_variables",
        "question": "How do I mark variables and outputs as sensitive in Terraform?",
        "expected_terms": ["sensitive", "true", "variable", "output", "masked"],
        "category": "security",
        "notes": "Sensitive data handling in Terraform",
    },
    {
        "id": "terraform_required_providers",
        "question": "What goes in the required_providers block and why is it important?",
        "expected_terms": ["required_providers", "source", "version", "terraform", "registry"],
        "category": "basics",
        "notes": "Provider version constraints",
    },
]


def clean_response(text: str) -> str:
    """Remove ANSI escape codes and control characters."""
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\[\?[0-9]+[hl]', '', text)
    return text.strip()


def query_model(model: str, prompt: str, timeout: int = 60) -> tuple[str, float]:
    """Query Ollama model with timeout."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return clean_response(result.stdout), elapsed
    except subprocess.TimeoutExpired:
        return f"TIMEOUT", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def calculate_term_coverage(response: str, expected_terms: list) -> tuple[float, list, list]:
    """Calculate what percentage of expected terms appear in response."""
    response_lower = response.lower()
    found = [t for t in expected_terms if t.lower() in response_lower]
    missing = [t for t in expected_terms if t.lower() not in response_lower]
    score = len(found) / len(expected_terms) if expected_terms else 0
    return score, found, missing


def run_comparison(base_model: str, finetuned_model: str):
    """Run comparison between base and fine-tuned models."""
    print("=" * 80)
    print("  Domain-Specific Terraform Knowledge Evaluation")
    print("=" * 80)
    print(f"\nBase Model:       {base_model}")
    print(f"Fine-tuned Model: {finetuned_model}")
    print(f"Questions:        {len(DOMAIN_QUESTIONS)}")
    print("=" * 80)

    results = []

    for i, q in enumerate(DOMAIN_QUESTIONS, 1):
        print(f"\n[{i}/{len(DOMAIN_QUESTIONS)}] {q['id']}")
        print(f"  Q: {q['question'][:70]}...")
        print(f"  Expected: {', '.join(q['expected_terms'][:5])}...")

        # Query both models
        print(f"  Querying base model...", end=" ", flush=True)
        base_resp, base_time = query_model(base_model, q['question'], timeout=90)
        base_score, base_found, base_missing = calculate_term_coverage(base_resp, q['expected_terms'])
        print(f"Done ({base_time:.1f}s)")

        print(f"  Querying fine-tuned...", end=" ", flush=True)
        ft_resp, ft_time = query_model(finetuned_model, q['question'], timeout=90)
        ft_score, ft_found, ft_missing = calculate_term_coverage(ft_resp, q['expected_terms'])
        print(f"Done ({ft_time:.1f}s)")

        # Determine winner
        if ft_score > base_score:
            winner = "FINE-TUNED"
        elif base_score > ft_score:
            winner = "BASE"
        else:
            winner = "TIE"

        print(f"  Base:       {base_score:.0%} ({len(base_found)}/{len(q['expected_terms'])} terms)")
        print(f"  Fine-tuned: {ft_score:.0%} ({len(ft_found)}/{len(q['expected_terms'])} terms)")
        print(f"  Winner:     {winner}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "base_score": base_score,
            "ft_score": ft_score,
            "base_found": base_found,
            "ft_found": ft_found,
            "base_missing": base_missing,
            "ft_missing": ft_missing,
            "base_time": base_time,
            "ft_time": ft_time,
            "base_length": len(base_resp),
            "ft_length": len(ft_resp),
            "winner": winner,
        })

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    base_avg = sum(r["base_score"] for r in results) / len(results)
    ft_avg = sum(r["ft_score"] for r in results) / len(results)

    wins = {"base": 0, "finetuned": 0, "tie": 0}
    for r in results:
        if r["winner"] == "BASE":
            wins["base"] += 1
        elif r["winner"] == "FINE-TUNED":
            wins["finetuned"] += 1
        else:
            wins["tie"] += 1

    print(f"\n{'Metric':<30} {'Base':>12} {'Fine-tuned':>12} {'Diff':>12}")
    print("-" * 68)
    print(f"{'Avg Term Coverage':<30} {base_avg:>11.1%} {ft_avg:>11.1%} {ft_avg-base_avg:>+11.1%}")
    print(f"{'Avg Response Time':<30} {sum(r['base_time'] for r in results)/len(results):>10.1f}s {sum(r['ft_time'] for r in results)/len(results):>10.1f}s")
    print(f"{'Wins':<30} {wins['base']:>12} {wins['finetuned']:>12}")
    print(f"{'Ties':<30} {wins['tie']:>12}")

    # Per-category breakdown
    print("\n\nPer-Category Results:")
    print("-" * 68)
    categories = set(r["category"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_base = sum(r["base_score"] for r in cat_results) / len(cat_results)
        cat_ft = sum(r["ft_score"] for r in cat_results) / len(cat_results)
        cat_wins = sum(1 for r in cat_results if r["winner"] == "FINE-TUNED")
        print(f"  {cat:<15} Base: {cat_base:.0%}  Fine-tuned: {cat_ft:.0%}  FT Wins: {cat_wins}/{len(cat_results)}")

    # Verdict
    print("\n" + "=" * 80)
    improvement = ft_avg - base_avg
    if improvement > 0.05:
        verdict = "Fine-tuning IMPROVED domain knowledge"
    elif improvement < -0.05:
        verdict = "Fine-tuning DEGRADED domain knowledge"
    else:
        verdict = "No significant difference in domain knowledge"
    print(f"  VERDICT: {verdict}")
    print(f"  Overall improvement: {improvement:+.1%}")
    print("=" * 80)

    # Save detailed results
    output_path = Path(__file__).parent / "domain_eval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "base_model": base_model,
            "finetuned_model": finetuned_model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "summary": {
                "base_avg": base_avg,
                "ft_avg": ft_avg,
                "improvement": improvement,
                "wins": wins,
            }
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_comparison("deepseek-coder:6.7b-instruct", "terraform-expert")
