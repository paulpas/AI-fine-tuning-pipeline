#!/usr/bin/env python3
"""
Fast dataset deduplication using hashing.
"""

import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict

INPUT_PATH = Path("/home/paulpas/git/ideas/llm_training_web_data/data/training/alpaca.json")
OUTPUT_PATH = Path("/home/paulpas/git/ideas/llm_training_web_data/data/training/alpaca_deduped.json")


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_hash(text: str) -> str:
    """Compute hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def count_repeated_phrases(text: str, min_len: int = 30) -> int:
    """Count repeated phrases in text."""
    text = normalize_text(text)

    # Split into chunks
    chunks = []
    for i in range(0, len(text) - min_len, min_len // 2):
        chunk = text[i:i + min_len]
        chunks.append(chunk)

    if not chunks:
        return 0

    # Count occurrences
    counts = defaultdict(int)
    for chunk in chunks:
        counts[chunk] += 1

    # Return max repetition count
    return max(counts.values()) if counts else 0


def main():
    print(f"Loading dataset from: {INPUT_PATH}")

    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)

    print(f"Original samples: {len(data):,}")

    # Track unique samples
    seen_instruction_hashes = set()
    seen_output_hashes = set()
    kept_samples = []

    stats = {
        "duplicate_instruction": 0,
        "duplicate_output": 0,
        "repetitive_output": 0,
        "short_output": 0,
        "kept": 0,
    }

    for i, sample in enumerate(data):
        if i % 10000 == 0:
            print(f"Processing {i:,}/{len(data):,}...")

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        # Skip very short outputs
        if len(output.strip()) < 50:
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

        # Check for repetitive patterns (repeated phrases)
        repetition_count = count_repeated_phrases(output)
        if repetition_count > 3:
            stats["repetitive_output"] += 1
            continue

        # Keep this sample
        seen_instruction_hashes.add(instruction_hash)
        seen_output_hashes.add(output_hash)
        kept_samples.append(sample)
        stats["kept"] += 1

    # Save deduplicated dataset
    print(f"\nSaving deduplicated dataset to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(kept_samples, f, indent=2)

    # Print stats
    print("\n" + "=" * 50)
    print("Deduplication Statistics")
    print("=" * 50)
    print(f"Original samples:       {len(data):,}")
    print(f"Duplicate instructions: {stats['duplicate_instruction']:,}")
    print(f"Duplicate outputs:      {stats['duplicate_output']:,}")
    print(f"Repetitive outputs:     {stats['repetitive_output']:,}")
    print(f"Short outputs:          {stats['short_output']:,}")
    print(f"Kept samples:           {stats['kept']:,}")
    print(f"Reduction:              {(1 - stats['kept']/len(data))*100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
