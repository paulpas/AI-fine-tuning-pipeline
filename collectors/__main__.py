#!/usr/bin/env python3
"""
Collector CLI

Run data collectors from the command line.

Usage:
    # List available collectors
    python -m collectors --list

    # Run a specific collector
    python -m collectors stackoverflow --output data/training/

    # Run multiple collectors
    python -m collectors stackoverflow github --output data/training/

    # Run all collectors
    python -m collectors --all --output data/training/

    # Show collector config options
    python -m collectors stackoverflow --help-config
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .registry import list_collectors, create_collector, get_collector


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def list_available_collectors():
    """Print list of available collectors"""
    collectors = list_collectors()

    print("\nAvailable Collectors:")
    print("=" * 60)

    for c in collectors:
        print(f"\n  {c['name']}")
        print(f"    {c['description']}")
        if c['required_config']:
            print(f"    Required config: {', '.join(c['required_config'])}")

    print("\n" + "=" * 60)
    print(f"Total: {len(collectors)} collectors\n")


def show_collector_config(name: str):
    """Show configuration options for a collector"""
    cls = get_collector(name)
    if not cls:
        print(f"Error: Unknown collector '{name}'")
        return

    # Create temporary instance
    instance = cls({})

    print(f"\nConfiguration for '{name}':")
    print("=" * 60)

    required = instance.get_required_config_keys()
    optional = instance.get_optional_config_keys()

    if required:
        print("\nRequired:")
        for key in required:
            print(f"  {key}")

    if optional:
        print("\nOptional (with defaults):")
        for key, default in optional.items():
            default_str = json.dumps(default) if isinstance(default, (list, dict)) else str(default)
            if len(default_str) > 50:
                default_str = default_str[:50] + "..."
            print(f"  {key}: {default_str}")

    print()


def load_env_config():
    """Load configuration from .env file"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def run_collector(
    name: str,
    output_dir: str,
    config: Optional[dict] = None,
    limit: Optional[int] = None,
    format: str = "alpaca"
) -> int:
    """Run a single collector"""
    logger = logging.getLogger("collectors")

    try:
        collector = create_collector(name, config or {})
        pairs = collector.run(output_dir, limit=limit, format=format)
        return len(pairs)
    except ImportError as e:
        logger.error(f"Missing dependency for '{name}': {e}")
        return 0
    except Exception as e:
        logger.error(f"Error running '{name}': {e}")
        return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Data Collectors for LLM Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           List available collectors
  %(prog)s stackoverflow -o data/           Run Stack Overflow collector
  %(prog)s github stackoverflow -o data/    Run multiple collectors
  %(prog)s --all -o data/                   Run all collectors
  %(prog)s stackoverflow --help-config      Show config options
        """
    )

    parser.add_argument(
        "collectors",
        nargs="*",
        help="Collectors to run (by name)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available collectors"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all available collectors"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/training",
        help="Output directory (default: data/training)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum items to collect per collector"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["alpaca", "sharegpt", "full"],
        default="alpaca",
        help="Output format (default: alpaca)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="JSON config file or inline JSON string"
    )
    parser.add_argument(
        "--help-config",
        action="store_true",
        help="Show config options for specified collector"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    load_env_config()

    # Handle --list
    if args.list:
        list_available_collectors()
        return 0

    # Handle --help-config
    if args.help_config:
        if not args.collectors:
            print("Error: Specify a collector name with --help-config")
            return 1
        show_collector_config(args.collectors[0])
        return 0

    # Determine which collectors to run
    if args.all:
        collectors_to_run = [c["name"] for c in list_collectors()]
    elif args.collectors:
        collectors_to_run = args.collectors
    else:
        parser.print_help()
        return 1

    # Load config
    config = {}
    if args.config:
        if args.config.startswith("{"):
            config = json.loads(args.config)
        elif Path(args.config).exists():
            with open(args.config) as f:
                config = json.load(f)
        else:
            print(f"Error: Config file not found: {args.config}")
            return 1

    # Run collectors
    logger = logging.getLogger("collectors")
    total_pairs = 0

    for name in collectors_to_run:
        logger.info(f"Running collector: {name}")
        count = run_collector(
            name,
            args.output,
            config=config,
            limit=args.limit,
            format=args.format
        )
        total_pairs += count
        logger.info(f"Collector '{name}' collected {count} pairs")

    logger.info(f"Total collected: {total_pairs} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
