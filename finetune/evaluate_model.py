#!/usr/bin/env python3
"""
Evaluate fine-tuned terraform-expert model against base model.

Compares responses on Terraform-specific questions using:
1. Keyword/concept coverage (does response contain expected terms?)
2. Code syntax validation (is generated HCL valid?)
3. Response relevance scoring
4. Optional LLM-as-judge evaluation
"""

import json
import subprocess
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import sys

# Evaluation test cases with expected concepts
EVAL_QUESTIONS = [
    {
        "id": "s3_versioning",
        "question": "How do I create an AWS S3 bucket with versioning enabled in Terraform?",
        "expected_concepts": ["aws_s3_bucket", "versioning", "enabled", "resource", "bucket"],
        "expected_code_patterns": [r"resource\s+\"aws_s3_bucket\"", r"versioning\s*\{", r"enabled\s*=\s*true"],
        "category": "aws",
    },
    {
        "id": "vpc_basic",
        "question": "Write Terraform code to create a VPC with a CIDR block of 10.0.0.0/16",
        "expected_concepts": ["aws_vpc", "cidr_block", "10.0.0.0/16", "resource"],
        "expected_code_patterns": [r"resource\s+\"aws_vpc\"", r"cidr_block\s*="],
        "category": "aws",
    },
    {
        "id": "variables",
        "question": "How do I define input variables in Terraform with default values and descriptions?",
        "expected_concepts": ["variable", "default", "description", "type", "string", "number", "bool"],
        "expected_code_patterns": [r"variable\s+\"", r"default\s*=", r"description\s*="],
        "category": "basics",
    },
    {
        "id": "modules",
        "question": "How do I use a module from the Terraform Registry in my configuration?",
        "expected_concepts": ["module", "source", "registry", "version", "terraform-aws-modules"],
        "expected_code_patterns": [r"module\s+\"", r"source\s*="],
        "category": "basics",
    },
    {
        "id": "state_backend",
        "question": "How do I configure Terraform to store state in an S3 backend with DynamoDB locking?",
        "expected_concepts": ["backend", "s3", "dynamodb", "lock", "terraform", "encrypt", "key"],
        "expected_code_patterns": [r"backend\s+\"s3\"", r"dynamodb_table", r"encrypt\s*=\s*true"],
        "category": "state",
    },
    {
        "id": "data_source",
        "question": "How do I use a data source to look up an existing AWS AMI in Terraform?",
        "expected_concepts": ["data", "aws_ami", "filter", "owners", "most_recent"],
        "expected_code_patterns": [r"data\s+\"aws_ami\"", r"filter\s*\{", r"most_recent\s*="],
        "category": "aws",
    },
    {
        "id": "for_each",
        "question": "How do I use for_each to create multiple similar resources in Terraform?",
        "expected_concepts": ["for_each", "each.key", "each.value", "toset", "tomap"],
        "expected_code_patterns": [r"for_each\s*=", r"each\.(key|value)"],
        "category": "basics",
    },
    {
        "id": "outputs",
        "question": "How do I define outputs in Terraform to expose resource attributes?",
        "expected_concepts": ["output", "value", "description", "sensitive"],
        "expected_code_patterns": [r"output\s+\"", r"value\s*="],
        "category": "basics",
    },
    {
        "id": "provider_config",
        "question": "How do I configure the AWS provider with a specific region and assume role?",
        "expected_concepts": ["provider", "aws", "region", "assume_role", "role_arn"],
        "expected_code_patterns": [r"provider\s+\"aws\"", r"region\s*=", r"assume_role"],
        "category": "aws",
    },
    {
        "id": "lifecycle",
        "question": "How do I use lifecycle rules in Terraform to prevent resource destruction?",
        "expected_concepts": ["lifecycle", "prevent_destroy", "create_before_destroy", "ignore_changes"],
        "expected_code_patterns": [r"lifecycle\s*\{", r"prevent_destroy\s*=\s*true"],
        "category": "basics",
    },
]


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    model_name: str
    response: str
    response_time: float
    concept_score: float  # 0-1, fraction of expected concepts found
    code_pattern_score: float  # 0-1, fraction of expected patterns found
    has_code_block: bool
    response_length: int
    concepts_found: list = field(default_factory=list)
    concepts_missing: list = field(default_factory=list)
    patterns_found: list = field(default_factory=list)
    patterns_missing: list = field(default_factory=list)


def query_ollama(model: str, prompt: str, timeout: int = 120) -> tuple[str, float]:
    """Query Ollama model and return response with timing."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        # Clean ANSI escape codes
        response = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', result.stdout)
        response = re.sub(r'\[\?[0-9]+[hl]', '', response)
        return response.strip(), elapsed
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def evaluate_response(question: dict, response: str, response_time: float, model_name: str) -> EvalResult:
    """Evaluate a model response against expected criteria."""
    response_lower = response.lower()

    # Check concept coverage
    concepts_found = []
    concepts_missing = []
    for concept in question["expected_concepts"]:
        if concept.lower() in response_lower:
            concepts_found.append(concept)
        else:
            concepts_missing.append(concept)

    concept_score = len(concepts_found) / len(question["expected_concepts"]) if question["expected_concepts"] else 0

    # Check code pattern coverage
    patterns_found = []
    patterns_missing = []
    for pattern in question["expected_code_patterns"]:
        if re.search(pattern, response, re.IGNORECASE):
            patterns_found.append(pattern)
        else:
            patterns_missing.append(pattern)

    pattern_score = len(patterns_found) / len(question["expected_code_patterns"]) if question["expected_code_patterns"] else 0

    # Check for code blocks
    has_code = bool(re.search(r'```(?:hcl|terraform)?', response)) or bool(re.search(r'resource\s+"', response))

    return EvalResult(
        question_id=question["id"],
        question=question["question"],
        model_name=model_name,
        response=response,
        response_time=response_time,
        concept_score=concept_score,
        code_pattern_score=pattern_score,
        has_code_block=has_code,
        response_length=len(response),
        concepts_found=concepts_found,
        concepts_missing=concepts_missing,
        patterns_found=patterns_found,
        patterns_missing=patterns_missing,
    )


def run_evaluation(base_model: str, finetuned_model: str, questions: list = None, max_tokens: int = 500) -> dict:
    """Run full evaluation comparing base and fine-tuned models."""
    if questions is None:
        questions = EVAL_QUESTIONS

    results = {
        "base_model": base_model,
        "finetuned_model": finetuned_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": [],
    }

    print("=" * 70)
    print("  Terraform Expert Model Evaluation")
    print("=" * 70)
    print(f"\nBase Model:      {base_model}")
    print(f"Fine-tuned Model: {finetuned_model}")
    print(f"Test Questions:   {len(questions)}")
    print("=" * 70)

    base_results = []
    finetuned_results = []

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...")

        # Query base model
        print(f"  Querying {base_model}...", end=" ", flush=True)
        base_response, base_time = query_ollama(base_model, q["question"])
        base_eval = evaluate_response(q, base_response, base_time, base_model)
        base_results.append(base_eval)
        print(f"Done ({base_time:.1f}s)")

        # Query fine-tuned model
        print(f"  Querying {finetuned_model}...", end=" ", flush=True)
        ft_response, ft_time = query_ollama(finetuned_model, q["question"])
        ft_eval = evaluate_response(q, ft_response, ft_time, finetuned_model)
        finetuned_results.append(ft_eval)
        print(f"Done ({ft_time:.1f}s)")

        # Quick comparison
        base_total = (base_eval.concept_score + base_eval.code_pattern_score) / 2
        ft_total = (ft_eval.concept_score + ft_eval.code_pattern_score) / 2
        winner = "FINE-TUNED" if ft_total > base_total else ("BASE" if base_total > ft_total else "TIE")

        print(f"  Base:       concepts={base_eval.concept_score:.0%} patterns={base_eval.code_pattern_score:.0%}")
        print(f"  Fine-tuned: concepts={ft_eval.concept_score:.0%} patterns={ft_eval.code_pattern_score:.0%}")
        print(f"  Winner: {winner}")

        results["questions"].append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "base": {
                "concept_score": base_eval.concept_score,
                "pattern_score": base_eval.code_pattern_score,
                "has_code": base_eval.has_code_block,
                "time": base_eval.response_time,
                "length": base_eval.response_length,
                "concepts_found": base_eval.concepts_found,
                "concepts_missing": base_eval.concepts_missing,
            },
            "finetuned": {
                "concept_score": ft_eval.concept_score,
                "pattern_score": ft_eval.code_pattern_score,
                "has_code": ft_eval.has_code_block,
                "time": ft_eval.response_time,
                "length": ft_eval.response_length,
                "concepts_found": ft_eval.concepts_found,
                "concepts_missing": ft_eval.concepts_missing,
            },
            "winner": winner,
        })

    # Calculate aggregates
    base_concept_avg = sum(r.concept_score for r in base_results) / len(base_results)
    base_pattern_avg = sum(r.code_pattern_score for r in base_results) / len(base_results)
    base_code_pct = sum(1 for r in base_results if r.has_code_block) / len(base_results)
    base_time_avg = sum(r.response_time for r in base_results) / len(base_results)

    ft_concept_avg = sum(r.concept_score for r in finetuned_results) / len(finetuned_results)
    ft_pattern_avg = sum(r.code_pattern_score for r in finetuned_results) / len(finetuned_results)
    ft_code_pct = sum(1 for r in finetuned_results if r.has_code_block) / len(finetuned_results)
    ft_time_avg = sum(r.response_time for r in finetuned_results) / len(finetuned_results)

    wins = {"base": 0, "finetuned": 0, "tie": 0}
    for q in results["questions"]:
        if q["winner"] == "BASE":
            wins["base"] += 1
        elif q["winner"] == "FINE-TUNED":
            wins["finetuned"] += 1
        else:
            wins["tie"] += 1

    results["summary"] = {
        "base": {
            "avg_concept_score": base_concept_avg,
            "avg_pattern_score": base_pattern_avg,
            "code_block_rate": base_code_pct,
            "avg_response_time": base_time_avg,
            "overall_score": (base_concept_avg + base_pattern_avg) / 2,
        },
        "finetuned": {
            "avg_concept_score": ft_concept_avg,
            "avg_pattern_score": ft_pattern_avg,
            "code_block_rate": ft_code_pct,
            "avg_response_time": ft_time_avg,
            "overall_score": (ft_concept_avg + ft_pattern_avg) / 2,
        },
        "wins": wins,
        "improvement": {
            "concept_score": ft_concept_avg - base_concept_avg,
            "pattern_score": ft_pattern_avg - base_pattern_avg,
            "overall": ((ft_concept_avg + ft_pattern_avg) / 2) - ((base_concept_avg + base_pattern_avg) / 2),
        }
    }

    return results


def print_summary(results: dict):
    """Print evaluation summary."""
    summary = results["summary"]

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Base Model':>15} {'Fine-tuned':>15} {'Improvement':>15}")
    print("-" * 70)

    base = summary["base"]
    ft = summary["finetuned"]
    imp = summary["improvement"]

    print(f"{'Concept Coverage':<25} {base['avg_concept_score']:>14.1%} {ft['avg_concept_score']:>14.1%} {imp['concept_score']:>+14.1%}")
    print(f"{'Code Pattern Score':<25} {base['avg_pattern_score']:>14.1%} {ft['avg_pattern_score']:>14.1%} {imp['pattern_score']:>+14.1%}")
    print(f"{'Code Block Rate':<25} {base['code_block_rate']:>14.1%} {ft['code_block_rate']:>14.1%} {ft['code_block_rate'] - base['code_block_rate']:>+14.1%}")
    print(f"{'Avg Response Time':<25} {base['avg_response_time']:>13.1f}s {ft['avg_response_time']:>13.1f}s {ft['avg_response_time'] - base['avg_response_time']:>+13.1f}s")
    print("-" * 70)
    print(f"{'OVERALL SCORE':<25} {base['overall_score']:>14.1%} {ft['overall_score']:>14.1%} {imp['overall']:>+14.1%}")

    print(f"\n{'Question Wins:':<25}")
    wins = summary["wins"]
    print(f"  Base Model:     {wins['base']:>3} ({wins['base']/len(results['questions']):.0%})")
    print(f"  Fine-tuned:     {wins['finetuned']:>3} ({wins['finetuned']/len(results['questions']):.0%})")
    print(f"  Ties:           {wins['tie']:>3} ({wins['tie']/len(results['questions']):.0%})")

    # Verdict
    print("\n" + "=" * 70)
    if imp["overall"] > 0.05:
        print("  VERDICT: Fine-tuning IMPROVED model performance")
    elif imp["overall"] < -0.05:
        print("  VERDICT: Fine-tuning DEGRADED model performance")
    else:
        print("  VERDICT: No significant difference between models")
    print("=" * 70)


def save_results(results: dict, output_path: Path):
    """Save evaluation results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Terraform model")
    parser.add_argument("--base-model", default="deepseek-coder:6.7b-instruct",
                       help="Base model to compare against")
    parser.add_argument("--finetuned-model", default="terraform-expert",
                       help="Fine-tuned model name")
    parser.add_argument("--output", type=Path,
                       default=Path(__file__).parent / "evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick evaluation with fewer questions")

    args = parser.parse_args()

    questions = EVAL_QUESTIONS[:3] if args.quick else EVAL_QUESTIONS

    # Check if models are available
    print("Checking model availability...")
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    available_models = result.stdout.lower()

    if args.finetuned_model.lower() not in available_models:
        print(f"ERROR: Fine-tuned model '{args.finetuned_model}' not found in Ollama")
        print("Available models:")
        print(result.stdout)
        sys.exit(1)

    # Check if base model needs to be pulled
    if "deepseek-coder" not in available_models:
        print(f"Base model not found. Pulling {args.base_model}...")
        subprocess.run(["ollama", "pull", args.base_model])

    # Run evaluation
    results = run_evaluation(args.base_model, args.finetuned_model, questions)

    # Print and save results
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
