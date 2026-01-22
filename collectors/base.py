"""
Base Collector Interface

All data collectors must inherit from BaseCollector and implement
the required abstract methods. This ensures a consistent interface
across all data sources.
"""

import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any


@dataclass
class QAPair:
    """
    A question-answer training pair.

    This is the standard format for all collectors. Each collector
    transforms its source data into QAPair objects.
    """
    instruction: str
    input: str
    output: str
    source: str  # e.g., "stackoverflow:12345" or "github:owner/repo:123"
    source_type: str  # e.g., "stackoverflow", "github", "web"
    score: int = 0
    tags: List[str] = field(default_factory=list)
    url: Optional[str] = None
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_alpaca(self) -> Dict:
        """Convert to Alpaca training format"""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }

    def to_sharegpt(self) -> Dict:
        """Convert to ShareGPT multi-turn format"""
        content = self.instruction
        if self.input:
            content += f"\n\n{self.input}"
        return {
            "conversations": [
                {"from": "human", "value": content},
                {"from": "gpt", "value": self.output}
            ]
        }

    def to_full(self) -> Dict:
        """Convert to full format with all metadata"""
        return asdict(self)

    def unique_id(self) -> str:
        """Generate a unique ID for deduplication"""
        content = f"{self.instruction}:{self.output}"
        return hashlib.md5(content.encode()).hexdigest()


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    Each collector must implement:
    - get_name(): Return the collector's unique name
    - get_description(): Return a human-readable description
    - collect(): Generator that yields raw data from the source
    - transform(): Convert raw data to QAPair objects

    Optional overrides:
    - validate_config(): Validate configuration
    - setup(): Initialize connections, authenticate, etc.
    - teardown(): Cleanup resources
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the collector.

        Args:
            config: Configuration dictionary. Keys depend on the collector.
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"collector.{self.get_name()}")
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for this collector"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s %(name)s [%(levelname)s] %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name of this collector.

        This is used for registration, CLI commands, and output file naming.
        Example: "stackoverflow", "github", "web_crawler"
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return a human-readable description of this collector.

        Example: "Collects Q&A from Stack Overflow via the API"
        """
        pass

    @abstractmethod
    def collect(self) -> Generator[Dict, None, None]:
        """
        Collect raw data from the source.

        Yields:
            Dict: Raw data items from the source. The structure
                  depends on the source (e.g., API response).
        """
        pass

    @abstractmethod
    def transform(self, raw_item: Dict) -> Optional[QAPair]:
        """
        Transform a raw data item into a QAPair.

        Args:
            raw_item: A single item yielded by collect()

        Returns:
            QAPair if transformation succeeds, None if item should be skipped
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the collector's configuration.

        Override this to check for required API keys, URLs, etc.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        return True

    def setup(self) -> None:
        """
        Initialize the collector.

        Override this to set up API clients, authenticate, etc.
        Called before collect() is invoked.
        """
        pass

    def teardown(self) -> None:
        """
        Clean up resources.

        Override this to close connections, flush buffers, etc.
        Called after collection is complete.
        """
        pass

    def run(
        self,
        output_dir: str,
        limit: Optional[int] = None,
        format: str = "alpaca"
    ) -> List[QAPair]:
        """
        Run the full collection pipeline.

        Args:
            output_dir: Directory to save output files
            limit: Maximum number of items to collect (None = no limit)
            format: Output format ("alpaca", "sharegpt", "full")

        Returns:
            List of collected QAPair objects
        """
        self.logger.info(f"Starting collection: {self.get_name()}")

        # Validate and setup
        self.validate_config()
        self.setup()

        pairs = []
        seen_ids = set()

        try:
            for i, raw_item in enumerate(self.collect()):
                if limit and i >= limit:
                    self.logger.info(f"Reached limit of {limit} items")
                    break

                try:
                    pair = self.transform(raw_item)
                    if pair:
                        # Deduplicate
                        uid = pair.unique_id()
                        if uid not in seen_ids:
                            seen_ids.add(uid)
                            pairs.append(pair)

                            if len(pairs) % 100 == 0:
                                self.logger.info(f"Collected {len(pairs)} pairs...")
                except Exception as e:
                    self.logger.warning(f"Transform error: {e}")
                    continue

        finally:
            self.teardown()

        self.logger.info(f"Collection complete: {len(pairs)} pairs")

        # Save output
        if pairs:
            self._save_output(pairs, output_dir, format)

        return pairs

    def _save_output(
        self,
        pairs: List[QAPair],
        output_dir: str,
        format: str
    ) -> None:
        """Save collected pairs to output files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        name = self.get_name()
        timestamp = datetime.now().strftime("%Y%m%d")

        # Convert to requested format
        if format == "alpaca":
            data = [p.to_alpaca() for p in pairs]
            filename = f"{name}_alpaca.json"
        elif format == "sharegpt":
            data = [p.to_sharegpt() for p in pairs]
            filename = f"{name}_sharegpt.json"
        else:  # full
            data = [p.to_full() for p in pairs]
            filename = f"{name}_full.json"

        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(pairs)} pairs to {filepath}")

    def get_required_config_keys(self) -> List[str]:
        """
        Return list of required configuration keys.

        Override this to specify required config (e.g., API keys).
        """
        return []

    def get_optional_config_keys(self) -> Dict[str, Any]:
        """
        Return dict of optional config keys with default values.
        """
        return {}
