#!/bin/bash
# Master Cloud/DevOps Engineering Questions - Model Evaluation Script
# Tests trained model on advanced topics NOT directly in training data
# Usage: ./test_master_questions.sh [model_name] [output_file]

set -e

MODEL="${1:-python-expert-v6}"
OUTPUT_FILE="${2:-test_results_$(date +%Y%m%d_%H%M%S).md}"
TEMPERATURE="0.7"
NUM_PREDICT="2048"

echo "Testing model: $MODEL"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if model exists in Ollama
if ! ollama list | grep -q "$MODEL"; then
    echo "Error: Model '$MODEL' not found in Ollama"
    echo "Available models:"
    ollama list
    exit 1
fi

# Initialize markdown output
cat > "$OUTPUT_FILE" << 'HEADER'
# Master Cloud/DevOps Engineering Questions - Model Evaluation

This document evaluates the model's knowledge on advanced cloud engineering topics.

**Model:**
**Date:**
**Temperature:** 0.7
**Context Length:** 2048 tokens

---

HEADER

echo "Starting evaluation..." | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Question 1: Python + pytest advanced patterns
echo "## Question 1: Python + pytest Advanced Patterns" >> "$OUTPUT_FILE"
Q1="Write a pytest fixture that implements a factory pattern for creating test fixtures with dependency injection. Include examples of how you'd use it to test async functions with mocking."
echo "**Question:** $Q1" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Answer:**" >> "$OUTPUT_FILE"
ollama run "$MODEL" --verbose=false "$Q1" 2>/dev/null | sed 's/^/> /' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Q1: $Q1"

# Question 2: AWS CLI advanced usage
echo "## Question 2: AWS CLI Advanced Usage Patterns" >> "$OUTPUT_FILE"
Q2="Explain how to use AWS CLI with STS assume-role to access multiple AWS accounts in a single script, including error handling and credential caching. Show how you'd structure this for CI/CD pipelines."
echo "**Question:** $Q2" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Answer:**" >> "$OUTPUT_FILE"
ollama run "$MODEL" --verbose=false "$Q2" 2>/dev/null | sed 's/^/> /' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Q2: $Q2"

# Question 3: Terraform complex AWS deployment
echo "## Question 3: Terraform - Complex AWS Deployment Architecture" >> "$OUTPUT_FILE"
Q3="Design a Terraform module architecture for deploying a production-grade multi-region active-active application on AWS with ELB, RDS with read replicas, ElastiCache, and cross-region replication. Include state management strategy and disaster recovery."
echo "**Question:** $Q3" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Answer:**" >> "$OUTPUT_FILE"
ollama run "$MODEL" --verbose=false "$Q3" 2>/dev/null | sed 's/^/> /' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Q3: $Q3"

# Question 4: Crossplane EKS cluster
echo "## Question 4: Crossplane - Complete EKS Cluster Provisioning" >> "$OUTPUT_FILE"
Q4="Create a Crossplane composition to provision a complete EKS cluster with all dependent AWS resources (VPC, subnets, security groups, IAM roles, node groups, OIDC provider). Include how you'd handle helm releases and addons through Crossplane."
echo "**Question:** $Q4" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Answer:**" >> "$OUTPUT_FILE"
ollama run "$MODEL" --verbose=false "$Q4" 2>/dev/null | sed 's/^/> /' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Q4: $Q4"

# Question 5: ArgoCD advanced patterns
echo "## Question 5: ArgoCD Advanced Usage - Kustomize + Helm Integration" >> "$OUTPUT_FILE"
Q5="Design an ArgoCD ApplicationSet that uses Kustomize overlays for multi-environment deployments with the Helm plugin for chart generation. Include git-ops workflow for progressive delivery with Kustomize patches and strategic merge patches."
echo "**Question:** $Q5" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Answer:**" >> "$OUTPUT_FILE"
ollama run "$MODEL" --verbose=false "$Q5" 2>/dev/null | sed 's/^/> /' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Q5: $Q5"

# Question 6: GatewayAPI with TLS
echo "## Question 6: Advanced GatewayAPI - TLS Proxying Configuration" >> "$OUTPUT_FILE"
Q6="Explain how to configure Kubernetes GatewayAPI with TLS termination, mTLS between gateway and backends, certificate rotation, and advanced routing based on SNI. Compare with traditional Ingress and Istio approaches."
echo "**Question:** $Q6" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Answer:**" >> "$OUTPUT_FILE"
ollama run "$MODEL" --verbose=false "$Q6" 2>/dev/null | sed 's/^/> /' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Q6: $Q6"

# Summary
echo "" >> "$OUTPUT_FILE"
echo "## Evaluation Summary" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "Model: $MODEL" >> "$OUTPUT_FILE"
echo "Questions asked: 6" >> "$OUTPUT_FILE"

echo ""
echo "✓ Evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_FILE"
echo "  # or with markdown viewer:"
echo "  mdcat $OUTPUT_FILE"
