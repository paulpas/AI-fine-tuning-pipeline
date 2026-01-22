#!/usr/bin/env python3
"""
Master Questions Evaluation - With/Without Web Search Comparison

Tests model answers on advanced topics:
1. Offline only (based on training data)
2. With web search context injected

This reveals gaps between training data and current best practices.
"""

import subprocess
import json
import sys
from datetime import datetime
from typing import Optional, Tuple
import re

# Web search would use: requests + BeautifulSoup, or curl, or your WebSearch API


QUESTIONS = [
    {
        "id": "python_pytest",
        "category": "Python + pytest",
        "question": """Write a pytest fixture that implements a factory pattern for creating test fixtures with dependency injection.
Include examples of how you'd use it to test async functions with mocking. Show best practices for fixture scope and cleanup.""",
        "search_query": "pytest fixture factory pattern dependency injection async 2024"
    },
    {
        "id": "aws_cli",
        "category": "AWS CLI Advanced",
        "question": """Explain how to use AWS CLI with STS assume-role to access multiple AWS accounts in a single script,
including error handling, credential caching, and session reuse. How would you structure this for CI/CD pipelines?""",
        "search_query": "AWS CLI STS assume-role multiple accounts CI/CD 2024"
    },
    {
        "id": "terraform_aws",
        "category": "Terraform Multi-Region AWS",
        "question": """Design a Terraform module architecture for a production-grade multi-region active-active application on AWS
with ELB, RDS read replicas, ElastiCache, and cross-region replication. Include state management and disaster recovery strategies.""",
        "search_query": "Terraform multi-region AWS active-active RDS replication 2024"
    },
    {
        "id": "crossplane_eks",
        "category": "Crossplane EKS",
        "question": """Create a Crossplane composition to provision a complete EKS cluster with all dependent AWS resources
(VPC, subnets, security groups, IAM roles, node groups, OIDC provider). How would you handle helm releases and addons?""",
        "search_query": "Crossplane EKS provisioning composition VPC IAM OIDC 2024"
    },
    {
        "id": "argocd_advanced",
        "category": "ArgoCD Advanced",
        "question": """Design an ArgoCD ApplicationSet using Kustomize overlays for multi-environment deployments with the Helm plugin
for chart generation. How would you implement progressive delivery with Kustomize patches and strategic merge patches?""",
        "search_query": "ArgoCD ApplicationSet Kustomize Helm plugin multi-environment 2024"
    },
    {
        "id": "gatewayapi_tls",
        "category": "GatewayAPI TLS",
        "question": """Explain advanced Kubernetes GatewayAPI configuration with TLS termination, mTLS between gateway and backends,
certificate rotation, and routing based on SNI. Compare with traditional Ingress and Istio approaches.""",
        "search_query": "Kubernetes GatewayAPI TLS mTLS SNI certificate rotation 2024"
    }
]


def query_ollama(model: str, prompt: str) -> str:
    """Query local Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, "--verbose=false"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT - response too long]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def get_web_context(query: str) -> Optional[str]:
    """
    Fetch web search context using DuckDuckGo (no API key required).

    Uses curl + jq to query DuckDuckGo's instant answer API.
    """
    try:
        # URL-encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)

        # Query DuckDuckGo API
        cmd = f"""curl -s "https://api.duckduckgo.com/?q={encoded_query}&format=json" | jq -r '.AbstractText, .RelatedTopics[0:3][] | .Text' 2>/dev/null"""

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            print(f"  [Web search returned empty result for: {query}]", file=sys.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"  [Web search timeout for: {query}]", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [Web search error: {e}]", file=sys.stderr)
        return None


def test_question(base_model: Optional[str], fine_tuned_model: str, question_data: dict, include_web: bool = True) -> dict:
    """Test a single question comparing base and fine-tuned models with/without web."""

    q_id = question_data["id"]
    category = question_data["category"]
    question = question_data["question"]
    search_query = question_data["search_query"]

    print(f"\n{'='*70}")
    print(f"Question: {category}")
    print(f"{'='*70}")

    result = {
        "id": q_id,
        "category": category,
        "question": question,
        "base_answer": None,
        "base_with_web_answer": None,
        "fine_tuned_answer": None,
        "fine_tuned_with_web_answer": None,
        "web_context": None
    }

    # Get base model answer (if provided)
    if base_model:
        print("  Querying BASE model (offline)...", file=sys.stderr)
        result["base_answer"] = query_ollama(base_model, question)

    # Get fine-tuned model answer (offline)
    print("  Querying FINE-TUNED model (offline)...", file=sys.stderr)
    result["fine_tuned_answer"] = query_ollama(fine_tuned_model, question)

    # Optionally get web context and requery both models
    if include_web:
        print("  Fetching web context...", file=sys.stderr)
        web_context = get_web_context(search_query)

        if web_context:
            result["web_context"] = web_context

            # Re-query base model with web context
            if base_model:
                enhanced_prompt = f"""Based on the following current information:

{web_context}

Please answer this question:

{question}"""

                print("  Querying BASE model (with web context)...", file=sys.stderr)
                result["base_with_web_answer"] = query_ollama(base_model, enhanced_prompt)

            # Re-query fine-tuned model with web context
            enhanced_prompt = f"""Based on the following current information:

{web_context}

Please answer this question:

{question}"""

            print("  Querying FINE-TUNED model (with web context)...", file=sys.stderr)
            result["fine_tuned_with_web_answer"] = query_ollama(fine_tuned_model, enhanced_prompt)

    return result


def generate_report(results: list, base_model: Optional[str], fine_tuned_model: str, output_file: str):
    """Generate markdown comparison report and write to both file and stdout."""

    def write_both(text: str):
        """Write to both file and stdout."""
        print(text)
        f.write(text + "\n")

    with open(output_file, "w") as f:
        # Header
        header = f"""# Master Cloud/DevOps Questions - Model Comparison
**Fine-tuned Model:** {fine_tuned_model}
**Base Model:** {base_model if base_model else 'None (offline only)'}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Questions:** {len(results)}
**Web Search:** {'Enabled' if results[0].get('web_context') else 'Disabled'}

---
"""
        write_both(header)

        for i, result in enumerate(results, 1):
            q_header = f"\n## Q{i}: {result['category']}\n\n**Question:**\n{result['question']}\n"
            write_both(q_header)

            # 4-way comparison table
            write_both("### Side-by-Side Comparison\n\n")

            if result['base_answer']:
                write_both("| Metric | Base (Offline) | Base (+ Web) | Fine-tuned (Offline) | Fine-tuned (+ Web) |")
                write_both("|--------|----------------|--------------|----------------------|---------------------|")
                write_both(f"| Length | {len(result['base_answer'])} | {len(result.get('base_with_web_answer', '')) or 'N/A'} | {len(result['fine_tuned_answer'])} | {len(result.get('fine_tuned_with_web_answer', '')) or 'N/A'} |")
                write_both(f"| Chars available | ✓ | {'✓' if result.get('base_with_web_answer') else '–'} | ✓ | {'✓' if result.get('fine_tuned_with_web_answer') else '–'} |")
                write_both("")
            else:
                write_both("| Metric | Fine-tuned (Offline) | Fine-tuned (+ Web) |")
                write_both("|--------|----------------------|---------------------|")
                write_both(f"| Length | {len(result['fine_tuned_answer'])} | {len(result.get('fine_tuned_with_web_answer', '')) or 'N/A'} |")
                write_both(f"| Available | ✓ | {'✓' if result.get('fine_tuned_with_web_answer') else '–'} |")
                write_both("")

            # Web context (if available)
            if result['web_context']:
                write_both("**Web Context Used:**\n")
                write_both(f"```\n{result['web_context'][:500]}\n```\n")

            # Base model answers
            if result['base_answer']:
                write_both("#### Base Model (Offline)\n")
                write_both(f"```\n{result['base_answer']}\n```\n")

                if result.get('base_with_web_answer'):
                    write_both("#### Base Model (+ Web Context)\n")
                    write_both(f"```\n{result['base_with_web_answer']}\n```\n")

            # Fine-tuned model answers
            write_both("#### Fine-tuned Model (Offline)\n")
            write_both(f"```\n{result['fine_tuned_answer']}\n```\n")

            if result.get('fine_tuned_with_web_answer'):
                write_both("#### Fine-tuned Model (+ Web Context)\n")
                write_both(f"```\n{result['fine_tuned_with_web_answer']}\n```\n")

                write_both("**Key Observations:**\n")
                write_both("- **Base vs Fine-tuned:** How much training improved specialized knowledge\n")
                write_both("- **Offline vs Web:** How much current information improves both models\n")
                write_both("- **Combined effect:** Best case = fine-tuned + web context\n\n")

            write_both("---\n")

        # Summary
        write_both("\n## Summary\n")
        write_both(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if base_model:
            write_both("### Four-Way Analysis\n\n")
            write_both("1. **Base (Offline)** - Baseline model without fine-tuning or web context\n")
            write_both("2. **Base (+ Web)** - Baseline improved with current web information\n")
            write_both("3. **Fine-tuned (Offline)** - Specialized model from training data only\n")
            write_both("4. **Fine-tuned (+ Web)** - Best case: specialized + current knowledge\n\n")
            write_both("**Metrics:**\n")
            write_both("- **Training Impact:** (Fine-tuned Offline vs Base Offline) = specialization gain\n")
            write_both("- **Web Impact:** (+ Web vs Offline) = real-time knowledge improvement\n")
            write_both("- **Combined Benefit:** (Fine-tuned + Web vs Base Offline) = total advantage\n")
        else:
            write_both("### Two-Way Analysis\n\n")
            write_both("1. **Fine-tuned (Offline)** - Specialized model from training data\n")
            write_both("2. **Fine-tuned (+ Web)** - Specialized model with current knowledge\n\n")
            write_both("**Metric:**\n")
            write_both("- **Web Impact:** (+Web vs Offline) = real-time knowledge improvement\n")


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Master questions evaluation with base/fine-tuned/web comparison"
    )
    parser.add_argument("--model", default="python-expert-v6", help="Fine-tuned Ollama model name")
    parser.add_argument("--base-model", help="Base model name (auto-detected from config if not provided)")
    parser.add_argument("--output", help="Output markdown file")
    parser.add_argument("--web", action="store_true", help="Include web search")
    parser.add_argument("--questions", type=int, help="Only test first N questions")

    args = parser.parse_args()

    # Auto-detect base model from config if not provided
    base_model = args.base_model
    if not base_model and not args.base_model:
        try:
            with open("config/pipeline_config.yaml") as f:
                config = yaml.safe_load(f)
                base_model = config.get("training", {}).get("base_model")
                if base_model:
                    # Try to find base model in Ollama (might not be available)
                    print(f"Auto-detected base model: {base_model}", file=sys.stderr)
        except Exception as e:
            print(f"Could not auto-detect base model: {e}", file=sys.stderr)

    # Verify model exists
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if args.model not in result.stdout:
            print(f"Error: Model '{args.model}' not found in Ollama", file=sys.stderr)
            print("\nAvailable models:", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error checking Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    # Select questions
    questions = QUESTIONS
    if args.questions:
        questions = questions[:args.questions]

    # Run tests
    print(f"Fine-tuned Model: {args.model}")
    print(f"Base Model: {base_model if base_model else 'None (offline comparison only)'}")
    print(f"Questions: {len(questions)}")
    print(f"Web search: {'enabled' if args.web else 'disabled'}")
    print()

    results = []
    for q_data in questions:
        try:
            result = test_question(base_model, args.model, q_data, include_web=args.web)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user", file=sys.stderr)
            break
        except Exception as e:
            print(f"Error testing question: {e}", file=sys.stderr)

    # Generate report
    output_file = args.output or f"evaluation_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_report(results, base_model, args.model, output_file)

    print(f"\n✓ Evaluation complete!")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
