"""
Unified LLM Training Pipeline

A complete pipeline for fine-tuning language models on code and documentation.

Usage:
    # Run full pipeline
    python -m pipeline.runner --config config/pipeline_config.yaml

    # Run specific stages
    python -m pipeline.runner --stage train,export

    # Use a profile
    python -m pipeline.runner --profile test

Modules:
    config_loader  - Load and validate pipeline configuration
    data_extractor - Extract training data from code/docs
    data_processor - Deduplicate and filter datasets
    model_exporter - Export trained models to Ollama
    runner         - Orchestrate pipeline stages
"""

__version__ = "1.0.0"
