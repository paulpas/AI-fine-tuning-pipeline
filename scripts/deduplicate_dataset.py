#!/usr/bin/env python3
"""
Fast dataset deduplication using hashing.
Removes duplicates, short outputs, and repetitive patterns.
"""

import json
import hashlib
import re
import argparse
from pathlib import Path
from collections import defaultdict


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


def deduplicate_dataset(
    input_path: str,
    output_path: str,
    min_output_length: int = 50,
    max_repetition: int = 3
) -> dict:
    """
    Deduplicate dataset and remove low-quality samples.

    Args:
        input_path: Path to input JSON dataset
        output_path: Path for output JSON dataset
        min_output_length: Minimum output length to keep
        max_repetition: Maximum allowed phrase repetitions

    Returns:
        Statistics dictionary
    """
    print(f"Loading dataset from: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    print(f"Original samples: {len(data):,}")

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
        "kept": 0,
    }

    for i, sample in enumerate(data):
        if i % 10000 == 0 and i > 0:
            print(f"Processing {i:,}/{len(data):,}...")

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

        # Keep this sample
        seen_instruction_hashes.add(instruction_hash)
        seen_output_hashes.add(output_hash)
        kept_samples.append(sample)
        stats["kept"] += 1

    # Save deduplicated dataset
    print(f"\nSaving deduplicated dataset to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept_samples, f, indent=2)

    # Print stats
    print("\n" + "=" * 50)
    print("Deduplication Statistics")
    print("=" * 50)
    print(f"Original samples:       {stats['original']:,}")
    print(f"Duplicate instructions: {stats['duplicate_instruction']:,}")
    print(f"Duplicate outputs:      {stats['duplicate_output']:,}")
    print(f"Repetitive outputs:     {stats['repetitive_output']:,}")
    print(f"Short outputs:          {stats['short_output']:,}")
    print(f"Kept samples:           {stats['kept']:,}")
    print(f"Reduction:              {(1 - stats['kept']/stats['original'])*100:.1f}%")
    print("=" * 50)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate training dataset")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--min-output-length", type=int, default=50,
                        help="Minimum output length to keep")
    parser.add_argument("--max-repetition", type=int, default=3,
                        help="Maximum allowed phrase repetitions")

    args = parser.parse_args()

    deduplicate_dataset(
        args.input,
        args.output,
        args.min_output_length,
        args.max_repetition
    )
