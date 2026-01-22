"""
Collector Registry

Provides automatic discovery and registration of collectors.
Use the @register_collector decorator to add new collectors.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Type, Optional

from .base import BaseCollector


# Global registry of collectors
_COLLECTORS: Dict[str, Type[BaseCollector]] = {}


def register_collector(cls: Type[BaseCollector]) -> Type[BaseCollector]:
    """
    Decorator to register a collector class.

    Usage:
        @register_collector
        class MyCollector(BaseCollector):
            ...
    """
    # Instantiate temporarily to get the name
    instance = cls.__new__(cls)
    instance.config = {}
    name = instance.get_name()

    if name in _COLLECTORS:
        raise ValueError(f"Collector '{name}' is already registered")

    _COLLECTORS[name] = cls
    return cls


def get_collector(name: str) -> Optional[Type[BaseCollector]]:
    """
    Get a collector class by name.

    Args:
        name: The collector's registered name

    Returns:
        The collector class, or None if not found
    """
    _discover_collectors()
    return _COLLECTORS.get(name)


def list_collectors() -> List[Dict]:
    """
    List all registered collectors.

    Returns:
        List of dicts with 'name', 'description', and 'class' keys
    """
    _discover_collectors()

    collectors = []
    for name, cls in sorted(_COLLECTORS.items()):
        instance = cls.__new__(cls)
        instance.config = {}
        collectors.append({
            "name": name,
            "description": instance.get_description(),
            "class": cls,
            "required_config": instance.get_required_config_keys(),
        })

    return collectors


def _discover_collectors():
    """
    Auto-discover collector modules in this package.

    This imports all Python files in the collectors/ directory,
    which triggers the @register_collector decorators.
    """
    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name not in ('__init__', 'base', 'registry'):
            try:
                importlib.import_module(f".{module_info.name}", package="collectors")
            except ImportError as e:
                # Log but don't fail - some collectors may have optional deps
                import logging
                logging.getLogger("collectors.registry").warning(
                    f"Could not import collector module '{module_info.name}': {e}"
                )


def create_collector(name: str, config: Optional[Dict] = None) -> BaseCollector:
    """
    Create a collector instance by name.

    Args:
        name: The collector's registered name
        config: Configuration dict to pass to the collector

    Returns:
        Instantiated collector

    Raises:
        ValueError: If collector not found
    """
    cls = get_collector(name)
    if cls is None:
        available = ", ".join(_COLLECTORS.keys())
        raise ValueError(f"Unknown collector: {name}. Available: {available}")

    return cls(config=config)
