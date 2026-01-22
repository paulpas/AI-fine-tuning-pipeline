"""
Data Processing Utilities

Handles deduplication and data quality filtering.

Replaces: scripts/deduplicate_dataset.py, finetune/deduplicate_dataset.py

Usage:
    from pipeline.data_processor import deduplicate_dataset

    stats = deduplicate_dataset(
        input_path="data/training/combined.json",
        output_path="data/training/deduped.json",
        min_output_length=50,
        max_repetition=3,
    )
"""

import json
import hashlib
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_hash(text: str) -> str:
    """Compute hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def count_repeated_phrases(text: str, min_len: int = 30) -> int:
    """Count repeated phrases in text (indicates repetitive content)."""
    text = normalize_text(text)

    # Split into overlapping chunks
    chunks = []
    for i in range(0, len(text) - min_len, min_len // 2):
        chunks.append(text[i:i + min_len])

    if not chunks:
        return 0

    # Count occurrences
    counts = defaultdict(int)
    for chunk in chunks:
        counts[chunk] += 1

    return max(counts.values()) if counts else 0


def has_repetitive_sentences(text: str, threshold: int = 3) -> bool:
    """Check if text has repetitive sentences."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2:
        return False

    counts = defaultdict(int)
    for s in sentences:
        normalized = normalize_text(s)
        counts[normalized] += 1

    return any(count >= threshold for count in counts.values())


def deduplicate_dataset(
    input_path: str,
    output_path: str,
    min_output_length: int = 50,
    max_repetition: int = 3,
    remove_similar: bool = False,
    similarity_threshold: float = 0.85,
) -> Dict[str, int]:
    """
    Deduplicate dataset and remove low-quality samples.

    Args:
        input_path: Path to input JSON dataset
        output_path: Path for output JSON dataset
        min_output_length: Minimum output length to keep
        max_repetition: Maximum allowed phrase repetitions
        remove_similar: Also remove near-duplicates (slower)
        similarity_threshold: Similarity threshold for near-duplicates

    Returns:
        Statistics dictionary
    """
    log.info(f"Loading dataset from: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    log.info(f"Original samples: {len(data):,}")

    # Track unique samples
    seen_instruction_hashes = set()
    seen_output_hashes = set()
    kept_samples = []

    stats = {
        "original": len(data),
        "duplicate_instruction": 0,
        "duplicate_output": 0,
        "repetitive_output": 0,
        "short_output": 0,
        "near_duplicate": 0,
        "kept": 0,
    }

    # For similarity checking
    recent_outputs = [] if remove_similar else None

    for i, sample in enumerate(data):
        if i % 10000 == 0 and i > 0:
            log.info(f"Processing {i:,}/{len(data):,}...")

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        # Skip very short outputs
        if len(output.strip()) < min_output_length:
            stats["short_output"] += 1
            continue

        # Hash instruction+input combo
        instruction_hash = compute_hash(instruction + " " + input_text)

        # Skip duplicate instructions
        if instruction_hash in seen_instruction_hashes:
            stats["duplicate_instruction"] += 1
            continue

        # Hash output
        output_hash = compute_hash(output)

        # Skip duplicate outputs
        if output_hash in seen_output_hashes:
            stats["duplicate_output"] += 1
            continue

        # Check for repetitive patterns
        repetition_count = count_repeated_phrases(output)
        if repetition_count > max_repetition:
            stats["repetitive_output"] += 1
            continue

        # Check for repetitive sentences
        if has_repetitive_sentences(output, threshold=max_repetition):
            stats["repetitive_output"] += 1
            continue

        # Near-duplicate check (optional, slower)
        if remove_similar and recent_outputs:
            from difflib import SequenceMatcher

            is_similar = False
            check_range = min(100, len(recent_outputs))
            for prev_output in recent_outputs[-check_range:]:
                if SequenceMatcher(None, output.lower(), prev_output.lower()).ratio() > similarity_threshold:
                    is_similar = True
                    stats["near_duplicate"] += 1
                    break

            if is_similar:
                continue

            recent_outputs.append(output)

        # Keep this sample
        seen_instruction_hashes.add(instruction_hash)
        seen_output_hashes.add(output_hash)
        kept_samples.append(sample)
        stats["kept"] += 1

    # Save deduplicated dataset
    log.info(f"Saving deduplicated dataset to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept_samples, f, indent=2)

    # Log stats
    log.info("")
    log.info("=" * 50)
    log.info("Deduplication Statistics")
    log.info("=" * 50)
    log.info(f"Original samples:       {stats['original']:,}")
    log.info(f"Duplicate instructions: {stats['duplicate_instruction']:,}")
    log.info(f"Duplicate outputs:      {stats['duplicate_output']:,}")
    log.info(f"Repetitive outputs:     {stats['repetitive_output']:,}")
    log.info(f"Short outputs:          {stats['short_output']:,}")
    if remove_similar:
        log.info(f"Near duplicates:        {stats['near_duplicate']:,}")
    log.info(f"Kept samples:           {stats['kept']:,}")
    reduction = (1 - stats['kept'] / stats['original']) * 100 if stats['original'] > 0 else 0
    log.info(f"Reduction:              {reduction:.1f}%")
    log.info("=" * 50)

    return stats


def filter_by_quality(
    input_path: str,
    output_path: str,
    min_instruction_length: int = 10,
    max_instruction_length: int = 500,
    min_output_length: int = 50,
    max_output_length: int = 10000,
) -> Dict[str, int]:
    """
    Filter dataset by quality metrics.

    Args:
        input_path: Path to input dataset
        output_path: Path for output dataset
        min_instruction_length: Minimum instruction length
        max_instruction_length: Maximum instruction length
        min_output_length: Minimum output length
        max_output_length: Maximum output length

    Returns:
        Statistics dictionary
    """
    log.info(f"Loading dataset from: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    stats = {
        "original": len(data),
        "instruction_too_short": 0,
        "instruction_too_long": 0,
        "output_too_short": 0,
        "output_too_long": 0,
        "kept": 0,
    }

    kept_samples = []

    for sample in data:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")

        if len(instruction) < min_instruction_length:
            stats["instruction_too_short"] += 1
            continue
        if len(instruction) > max_instruction_length:
            stats["instruction_too_long"] += 1
            continue
        if len(output) < min_output_length:
            stats["output_too_short"] += 1
            continue
        if len(output) > max_output_length:
            stats["output_too_long"] += 1
            continue

        kept_samples.append(sample)
        stats["kept"] += 1

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept_samples, f, indent=2)

    log.info(f"Kept {stats['kept']:,} of {stats['original']:,} samples")

    return stats


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate training dataset")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--min-output-length", type=int, default=50,
                        help="Minimum output length to keep")
    parser.add_argument("--max-repetition", type=int, default=3,
                        help="Maximum allowed phrase repetitions")
    parser.add_argument("--remove-similar", action="store_true",
                        help="Also remove near-duplicates (slower)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Similarity threshold for near-duplicates")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    deduplicate_dataset(
        args.input,
        args.output,
        args.min_output_length,
        args.max_repetition,
        args.remove_similar,
        args.similarity_threshold,
    )
