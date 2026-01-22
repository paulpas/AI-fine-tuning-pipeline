"""
Modular Data Collectors for LLM Training

This package provides a pluggable architecture for collecting training data
from various sources. Each collector is a separate module that can be
independently configured and run.

Architecture:
    collectors/
    ├── base.py              # Abstract base class for all collectors
    ├── registry.py          # Collector registration and discovery
    ├── stackoverflow.py     # Stack Exchange API collector
    ├── github.py            # GitHub API collector
    ├── web_crawler.py       # Generic web crawler with JS support
    └── __init__.py          # This file

Usage:
    # Run a specific collector
    python -m collectors.stackoverflow --output data/training/

    # Run all enabled collectors
    python -m collectors --all --output data/training/

    # List available collectors
    python -m collectors --list

Adding a new collector:
    1. Create a new file in collectors/ (e.g., my_source.py)
    2. Inherit from BaseCollector
    3. Implement required methods: collect(), transform(), get_name()
    4. Register with @register_collector decorator
"""

from .base import BaseCollector, QAPair
from .registry import register_collector, get_collector, list_collectors, create_collector

__all__ = [
    "BaseCollector",
    "QAPair",
    "register_collector",
    "get_collector",
    "list_collectors",
    "create_collector",
]

__version__ = "1.0.0"
