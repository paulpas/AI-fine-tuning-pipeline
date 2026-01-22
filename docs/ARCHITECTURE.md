# Pipeline Architecture & Tool Relationships

This document explains how each tool in the LLM fine-tuning pipeline feeds into the next.

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   GITHUB REPOS ─────┐                                                               │
│   (Source Code)     │              TRAINING DATA                    FINE-TUNED     │
│                     ├────────────▶ (JSON Files)   ──────────────▶   MODEL          │
│   WEB PAGES ────────┘                                                               │
│   (JS-rendered)                                                                     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Two Data Collection Paths

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   PATH A: GIT REPOSITORIES              PATH B: WEB PAGES                           │
│   ─────────────────────────             ───────────────────                         │
│                                                                                     │
│   ./scripts/collect_all_data.sh         python run_pipeline.py                      │
│                                                                                     │
│   ┌─────────────┐                       ┌─────────────┐                             │
│   │ git clone   │                       │ Playwright  │  ◄── JS rendering          │
│   └──────┬──────┘                       │ (crawler)   │                             │
│          │                              └──────┬──────┘                             │
│          ▼                                     │                                    │
│   ┌─────────────┐                              ▼                                    │
│   │ AST parsing │                       ┌─────────────┐                             │
│   │ (.py files) │                       │ HTML → MD   │  ◄── readability + mdify   │
│   └──────┬──────┘                       │ (extractor) │                             │
│          │                              └──────┬──────┘                             │
│          │                                     │                                    │
│          └──────────────┬──────────────────────┘                                    │
│                         │                                                           │
│                         ▼                                                           │
│                  ┌─────────────┐                                                    │
│                  │ Alpaca JSON │  ◄── Common output format                         │
│                  │ (training)  │                                                    │
│                  └─────────────┘                                                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Tool Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: DATA COLLECTION                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────────────┐                                                           │
│  │  config/             │                                                           │
│  │  data_sources.yaml   │  ◄─── USER CONFIGURES: Repository URLs, subdirs, types   │
│  └──────────┬───────────┘                                                           │
│             │                                                                       │
│             │ reads                                                                 │
│             ▼                                                                       │
│  ┌──────────────────────┐       ┌─────────────────────────────────────────────┐    │
│  │  collect_all_data.sh │──────▶│  git clone --depth 1                        │    │
│  │  (orchestrator)      │       │  (downloads repos to repos/ directory)      │    │
│  └──────────┬───────────┘       └─────────────────────────────────────────────┘    │
│             │                                                                       │
│             │ calls                                                                 │
│             ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                         EXTRACTION SCRIPTS                                   │   │
│  │  ┌─────────────────────────────┐   ┌─────────────────────────────┐          │   │
│  │  │  prepare_k8s_python_data.py │   │  prepare_pytest_data.py     │          │   │
│  │  │                             │   │                             │          │   │
│  │  │  INPUT:  .py files          │   │  INPUT:  .rst files         │          │   │
│  │  │  METHOD: Python AST parsing │   │  METHOD: RST code-block     │          │   │
│  │  │  OUTPUT: instruction/output │   │          extraction         │          │   │
│  │  │          pairs              │   │  OUTPUT: instruction/output │          │   │
│  │  │                             │   │          pairs              │          │   │
│  │  └──────────────┬──────────────┘   └──────────────┬──────────────┘          │   │
│  │                 │                                  │                         │   │
│  └─────────────────┼──────────────────────────────────┼─────────────────────────┘   │
│                    │                                  │                             │
│                    └──────────────┬───────────────────┘                             │
│                                   │                                                 │
│                                   ▼                                                 │
│                    ┌──────────────────────────────┐                                 │
│                    │  data/training/*_alpaca.json │  ◄─── Per-repo training files  │
│                    │  (individual datasets)       │                                 │
│                    └──────────────┬───────────────┘                                 │
│                                   │                                                 │
└───────────────────────────────────┼─────────────────────────────────────────────────┘
                                    │
                                    │ combines
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2: DATA PROCESSING                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐    │
│  │  Python inline script (in collect_all_data.sh)                             │    │
│  │                                                                            │    │
│  │  • Reads all *_alpaca.json files                                           │    │
│  │  • Merges into single list                                                 │    │
│  │  • Outputs: all_python_combined.json                                       │    │
│  └──────────────────────────────────┬─────────────────────────────────────────┘    │
│                                     │                                               │
│                                     ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐    │
│  │  deduplicate_dataset.py                                                    │    │
│  │                                                                            │    │
│  │  INPUT:  all_python_combined.json                                          │    │
│  │                                                                            │    │
│  │  FILTERS:                                                                  │    │
│  │  ├── Remove duplicate instructions (hash-based)                           │    │
│  │  ├── Remove duplicate outputs (hash-based)                                │    │
│  │  ├── Remove outputs < min_length (default: 50 chars)                      │    │
│  │  └── Remove repetitive content (pattern detection)                        │    │
│  │                                                                            │    │
│  │  OUTPUT: all_python_deduped.json  ◄─── FINAL TRAINING DATASET             │    │
│  └──────────────────────────────────┬─────────────────────────────────────────┘    │
│                                     │                                               │
└─────────────────────────────────────┼───────────────────────────────────────────────┘
                                      │
                                      │ feeds into
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 3: MODEL TRAINING                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────────────┐                                                           │
│  │  axolotl_config_     │  ◄─── USER CONFIGURES:                                   │
│  │  k8s_python.yaml     │       • base_model (e.g., google/codegemma-2b)           │
│  └──────────┬───────────┘       • LoRA settings (rank, alpha, dropout)             │
│             │                   • Training params (epochs, lr, batch_size)          │
│             │                   • Dataset path: all_python_deduped.json             │
│             │ configures                                                            │
│             ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  accelerate launch -m axolotl.cli.train                                     │   │
│  │                                                                             │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │   │
│  │  │   HuggingFace   │    │    Axolotl      │    │   Accelerate    │         │   │
│  │  │   Transformers  │◄───│  (orchestrates) │◄───│  (multi-GPU)    │         │   │
│  │  │   (model code)  │    │                 │    │                 │         │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘         │   │
│  │           │                                                                 │   │
│  │           │ downloads                                                       │   │
│  │           ▼                                                                 │   │
│  │  ┌─────────────────┐                                                        │   │
│  │  │  Base Model     │  google/codegemma-2b from HuggingFace Hub             │   │
│  │  │  (2B params)    │                                                        │   │
│  │  └────────┬────────┘                                                        │   │
│  │           │                                                                 │   │
│  │           │ + LoRA adapters                                                 │   │
│  │           ▼                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  TRAINING LOOP                                                      │   │   │
│  │  │                                                                     │   │   │
│  │  │  for each epoch:                                                    │   │   │
│  │  │    for each batch in dataset:                                       │   │   │
│  │  │      1. Tokenize instruction + output                               │   │   │
│  │  │      2. Forward pass through model                                  │   │   │
│  │  │      3. Calculate loss (cross-entropy)                              │   │   │
│  │  │      4. Backward pass (gradients)                                   │   │   │
│  │  │      5. Update LoRA weights only (not base model)                   │   │   │
│  │  │      6. Log metrics every N steps                                   │   │   │
│  │  │      7. Save checkpoint every N steps                               │   │   │
│  │  │      8. Evaluate on validation set                                  │   │   │
│  │  │                                                                     │   │   │
│  │  └─────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                             │   │
│  └─────────────────────────────────────┬───────────────────────────────────────┘   │
│                                        │                                            │
│                                        │ outputs                                    │
│                                        ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  finetune/output/python-expert-v3-codegemma-2b/                             │   │
│  │                                                                             │   │
│  │  ├── adapter_config.json      (LoRA configuration)                         │   │
│  │  ├── adapter_model.safetensors (LoRA weights - small, ~50MB)               │   │
│  │  ├── tokenizer.json           (tokenizer config)                           │   │
│  │  ├── tokenizer_config.json                                                 │   │
│  │  └── checkpoint-*/            (intermediate checkpoints)                   │   │
│  │                                                                             │   │
│  └─────────────────────────────────────┬───────────────────────────────────────┘   │
│                                        │                                            │
└────────────────────────────────────────┼────────────────────────────────────────────┘
                                         │
                                         │ feeds into
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 4: MODEL CONVERSION                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  merge_lora.py                                                              │   │
│  │                                                                             │   │
│  │  INPUT:  Base model (google/codegemma-2b)                                   │   │
│  │        + LoRA adapter (adapter_model.safetensors)                           │   │
│  │                                                                             │   │
│  │  PROCESS: Merges LoRA weights into base model weights                       │   │
│  │           W_merged = W_base + (alpha/rank) * (A @ B)                        │   │
│  │                                                                             │   │
│  │  OUTPUT: python-expert-v3-merged/ (full merged model, ~4GB)                 │   │
│  └─────────────────────────────────────┬───────────────────────────────────────┘   │
│                                        │                                            │
│                                        ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  convert_to_gguf.py                                                         │   │
│  │                                                                             │   │
│  │  INPUT:  Merged HuggingFace model (safetensors format)                      │   │
│  │                                                                             │   │
│  │  PROCESS:                                                                   │   │
│  │  1. Load model weights                                                      │   │
│  │  2. Convert to GGUF format (llama.cpp compatible)                           │   │
│  │  3. Quantize weights (q4_k_m = 4-bit with k-means)                          │   │
│  │                                                                             │   │
│  │  QUANTIZATION OPTIONS:                                                      │   │
│  │  ├── q4_k_m  (~1.5GB)  - Smallest, good quality                            │   │
│  │  ├── q5_k_m  (~1.8GB)  - Balanced                                          │   │
│  │  ├── q8_0    (~2.5GB)  - Higher quality                                    │   │
│  │  └── f16     (~4GB)    - Full precision                                    │   │
│  │                                                                             │   │
│  │  OUTPUT: python-expert-v3-gguf/model-q4_k_m.gguf                            │   │
│  └─────────────────────────────────────┬───────────────────────────────────────┘   │
│                                        │                                            │
│                                        ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  create_ollama_model.py                                                     │   │
│  │                                                                             │   │
│  │  INPUT:  GGUF file                                                          │   │
│  │                                                                             │   │
│  │  CREATES Modelfile:                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  FROM ./model-q4_k_m.gguf                                           │   │   │
│  │  │  PARAMETER temperature 0.5                                          │   │   │
│  │  │  PARAMETER repeat_penalty 1.3                                       │   │   │
│  │  │  PARAMETER num_predict 512                                          │   │   │
│  │  │  SYSTEM "You are a Python expert..."                                │   │   │
│  │  └─────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                             │   │
│  │  RUNS: ollama create python-expert-v3 -f Modelfile                          │   │
│  │                                                                             │   │
│  │  OUTPUT: Model registered in Ollama                                         │   │
│  └─────────────────────────────────────┬───────────────────────────────────────┘   │
│                                        │                                            │
└────────────────────────────────────────┼────────────────────────────────────────────┘
                                         │
                                         │ available for
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 5: DEPLOYMENT & EVALUATION                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  ollama run python-expert-v3                                                │   │
│  │                                                                             │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │   │
│  │  │   User Prompt   │───▶│  Ollama Server  │───▶│  GGUF Model     │         │   │
│  │  │                 │    │  (llama.cpp)    │    │  (quantized)    │         │   │
│  │  └─────────────────┘    └─────────────────┘    └────────┬────────┘         │   │
│  │                                                          │                  │   │
│  │                                                          ▼                  │   │
│  │                                               ┌─────────────────┐           │   │
│  │                                               │   Response      │           │   │
│  │                                               │   (generated)   │           │   │
│  │                                               └─────────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  evaluate_model.py                                                          │   │
│  │                                                                             │   │
│  │  COMPARES:                                                                  │   │
│  │  ├── Base model (codegemma:2b)                                             │   │
│  │  └── Fine-tuned model (python-expert-v3)                                   │   │
│  │                                                                             │   │
│  │  METRICS:                                                                   │   │
│  │  ├── Response quality (manual review)                                      │   │
│  │  ├── Domain-specific accuracy                                              │   │
│  │  └── Response time                                                         │   │
│  │                                                                             │   │
│  │  OUTPUT: evaluation_results.json                                            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Tool Input/Output Summary

```
┌────────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│          TOOL              │           INPUT             │           OUTPUT            │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ data_sources.yaml          │ User configuration          │ Repository definitions      │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ collect_all_data.sh        │ data_sources.yaml           │ Cloned repos + JSON files   │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ prepare_k8s_python_data.py │ .py files (repos/)          │ *_alpaca.json               │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ prepare_pytest_data.py     │ .rst files (repos/)         │ *_alpaca.json               │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ deduplicate_dataset.py     │ all_python_combined.json    │ all_python_deduped.json     │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ axolotl_config.yaml        │ User configuration          │ Training parameters         │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ accelerate + axolotl       │ Config + deduped JSON       │ LoRA adapter weights        │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ merge_lora.py              │ Base model + LoRA adapter   │ Merged model (safetensors)  │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ convert_to_gguf.py         │ Merged model                │ Quantized GGUF file         │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ create_ollama_model.py     │ GGUF file                   │ Ollama model registration   │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ evaluate_model.py          │ Base + fine-tuned models    │ Evaluation metrics          │
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

## Data Format Transformations

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Python Source  │     │  Alpaca JSON    │     │  Tokenized      │
│                 │     │                 │     │  Sequences      │
│  def func():    │     │  {              │     │                 │
│    """doc"""    │ ──▶ │   "instruction" │ ──▶ │  [101, 2054,    │
│    return x     │     │   "input": ""   │     │   3045, 102,    │
│                 │     │   "output"      │     │   ...]          │
└─────────────────┘     │  }              │     └─────────────────┘
                        └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
   EXTRACTION             TRAINING DATA            MODEL INPUT
   (AST parsing)          (JSON format)           (tensor IDs)
```

## File System Layout

```
llm_training_web_data/
│
├── config/
│   └── data_sources.yaml        ◄── START HERE: Configure repos
│
├── repos/                        ◄── Cloned repositories (git ignored)
│   ├── k8s-python/
│   ├── pytest-repo/
│   ├── fastapi/
│   └── ...
│
├── data/
│   └── training/                 ◄── Generated training data
│       ├── k8s_python_alpaca.json
│       ├── pytest_alpaca.json
│       ├── all_python_combined.json
│       └── all_python_deduped.json   ◄── FINAL DATASET
│
├── scripts/
│   ├── collect_all_data.sh       ◄── Run this first
│   ├── prepare_k8s_python_data.py
│   ├── prepare_pytest_data.py
│   ├── deduplicate_dataset.py
│   ├── run_pipeline.sh           ◄── Or run this for everything
│   ├── merge_lora.py
│   ├── convert_to_gguf.py
│   └── create_ollama_model.py
│
├── finetune/
│   ├── axolotl_config_k8s_python.yaml  ◄── Training config
│   ├── pipeline_config.yaml
│   ├── training_v3.log           ◄── Training progress
│   └── output/                   ◄── Model outputs
│       └── python-expert-v3-codegemma-2b/
│           ├── adapter_model.safetensors
│           └── ...
│
└── docs/
    ├── DATA_COLLECTION.md
    └── ARCHITECTURE.md           ◄── THIS FILE
```

## Execution Order

```
    ┌─────┐
    │  1  │  Configure data_sources.yaml
    └──┬──┘
       │
       ▼
    ┌─────┐
    │  2  │  ./scripts/collect_all_data.sh
    └──┬──┘
       │
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
    ┌─────┐                               ┌─────┐
    │ 2a  │  git clone (repos)            │ 2b  │  Extract (prepare_*.py)
    └──┬──┘                               └──┬──┘
       │                                      │
       └──────────────────┬───────────────────┘
                          │
                          ▼
                       ┌─────┐
                       │ 2c  │  Combine + Deduplicate
                       └──┬──┘
                          │
                          ▼
                       ┌─────┐
                       │  3  │  Edit axolotl_config.yaml
                       └──┬──┘
                          │
                          ▼
                       ┌─────┐
                       │  4  │  accelerate launch -m axolotl.cli.train
                       └──┬──┘
                          │
                          ▼
                       ┌─────┐
                       │  5  │  merge_lora.py
                       └──┬──┘
                          │
                          ▼
                       ┌─────┐
                       │  6  │  convert_to_gguf.py
                       └──┬──┘
                          │
                          ▼
                       ┌─────┐
                       │  7  │  create_ollama_model.py
                       └──┬──┘
                          │
                          ▼
                       ┌─────┐
                       │  8  │  ollama run python-expert-v3
                       └─────┘
```

## Key Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PYTHON PACKAGES                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Training Stack:                    Conversion Stack:                               │
│  ├── torch (PyTorch)                ├── llama-cpp-python                           │
│  ├── transformers (HuggingFace)     ├── gguf                                       │
│  ├── accelerate (multi-GPU)         └── sentencepiece                              │
│  ├── axolotl (training framework)                                                  │
│  ├── peft (LoRA implementation)     Inference:                                     │
│  ├── bitsandbytes (quantization)    └── ollama (external binary)                   │
│  └── datasets (data loading)                                                       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
