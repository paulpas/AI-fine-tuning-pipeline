# Terraform Expert Fine-Tuning Evaluation Report

**Date:** 2026-01-12
**Base Model:** deepseek-coder-6.7b-instruct
**Fine-tuned Model:** terraform-expert
**Training Dataset:** 31,958 samples from HashiCorp Terraform documentation
**Training Duration:** ~6 hours (110 steps, 2 epochs)
**Final Training Loss:** 1.63 (from initial 5.24)

---

## Executive Summary

The fine-tuning produced **mixed results**. While the model shows improvements in certain areas (sensitive variables, required_providers, basics), it suffers from a critical issue: **repetitive output generation** that causes timeouts on many queries.

### Key Findings

| Metric | Base Model | Fine-tuned | Change |
|--------|------------|------------|--------|
| Avg Term Coverage | 75.6% | 52.7% | -22.9% |
| Response Time | 9.4s | 43.7s | +34.3s |
| Timeout Rate | 0% | 37.5% | +37.5% |
| Wins (when not timed out) | 3 | 3 | Tie |

---

## Detailed Results

### Questions Where Fine-Tuned Model Won

| Question | Base | Fine-tuned | Notes |
|----------|------|------------|-------|
| sensitive_variables | 40% | **80%** | +40% improvement |
| required_providers | 80% | **100%** | +20% improvement |
| moved_block | 50% | **67%** | +17% improvement |

### Questions Where Base Model Won

| Question | Base | Fine-tuned | Notes |
|----------|------|------------|-------|
| import_block | **100%** | 0% | Fine-tuned timed out |
| replace_triggered | **100%** | 0% | Fine-tuned timed out |
| check_block | **60%** | 0% | Fine-tuned timed out |

### Ties

| Question | Base | Fine-tuned |
|----------|------|------------|
| terraform_cloud | 100% | 100% |
| provider_meta | 75% | 75% |

---

## Root Cause Analysis

### The Repetition Problem

The fine-tuned model exhibits a known issue with LoRA fine-tuning: **output repetition loops**. When generating responses, the model sometimes enters a loop where it repeats similar phrases endlessly, causing:

1. Excessive generation time (90s+ timeouts)
2. Very long, repetitive responses
3. Loss of coherent answer structure

**Example from initial test (S3 bucket question):**
```
...It shows how to install a provider by downloading an archive containing
precompiled binaries (for example via a package manager) instead of building
from source using Go or another supported language. It also explains why
Terraform recommends using version control even without it; it is critical
when you are managing complex infrastructure deployments across multiple
people or teams where changes must be tracked. It shows how to install a
provider by downloading an archive...  [REPEATS]
```

### Likely Causes

1. **Training data quality**: Synthetic Q&A pairs may have introduced repetitive patterns
2. **Training hyperparameters**:
   - Learning rate (0.0002) may be too high
   - Only 2 epochs may not be enough for convergence
3. **LoRA rank/alpha ratio**: r=16, alpha=32 may need tuning
4. **No repetition penalty**: Model wasn't trained with repetition penalty

---

## Recommendations for Improvement

### Immediate Fixes (Inference Time)

1. **Add repetition penalty** to Ollama Modelfile:
   ```
   PARAMETER repeat_penalty 1.2
   PARAMETER repeat_last_n 64
   ```

2. **Reduce max tokens** to prevent runaway generation:
   ```
   PARAMETER num_predict 500
   ```

3. **Lower temperature** for more focused output:
   ```
   PARAMETER temperature 0.5
   ```

### Training Improvements (For Re-training)

1. **Dataset Curation**
   - Remove duplicate/near-duplicate training samples
   - Add more diverse instruction phrasings
   - Include explicit stop sequences in training data

2. **Hyperparameter Tuning**
   - Lower learning rate: 0.0001 → 0.00005
   - More epochs: 2 → 4-6
   - Smaller LoRA rank for more regularization: r=8

3. **Training Technique**
   - Add DPO (Direct Preference Optimization) stage
   - Use higher dropout: 0.05 → 0.1
   - Enable gradient checkpointing with longer sequences

4. **Evaluation During Training**
   - Add eval steps with repetition detection
   - Early stopping based on eval metrics

---

## Positive Indicators

Despite the repetition issue, when the model produces valid responses:

1. **Better coverage** on security-related concepts (+40%)
2. **Improved basics** knowledge (+20% on required_providers)
3. **More complete** answers (when not looping)
4. **Training loss convergence** indicates the model did learn

---

## Conclusion

**The fine-tuning partially succeeded** - the model learned domain knowledge from the HashiCorp documentation, as evidenced by improved scores on several questions. However, the training introduced a **repetition bug** that severely impacts practical usability.

### Verdict: PARTIAL SUCCESS with CRITICAL BUG

**Recommended Action:**
1. Apply inference-time fixes (repetition penalty) immediately
2. Re-train with improved hyperparameters and dataset curation
3. Consider alternative training approaches (full fine-tuning, DPO)

---

## Files Generated

| File | Description |
|------|-------------|
| `evaluation_results.json` | General evaluation results |
| `domain_eval_results.json` | Domain-specific evaluation results |
| `evaluate_model.py` | General evaluation script |
| `evaluate_domain_specific.py` | Domain-specific evaluation script |
