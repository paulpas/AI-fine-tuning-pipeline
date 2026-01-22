#!/usr/bin/env python3
"""
Deduplicate training dataset to reduce repetition issues.

Removes:
1. Exact duplicates
2. Near-duplicates (high similarity)
3. Samples with repetitive patterns
"""

import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

INPUT_PATH = Path("/home/paulpas/git/ideas/llm_training_web_data/data/training/alpaca.json")
OUTPUT_PATH = Path("/home/paulpas/git/ideas/llm_training_web_data/data/training/alpaca_deduped.json")


def compute_hash(text: str) -> str:
    """Compute hash of normalized text."""
    # Normalize: lowercase, remove extra whitespace
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def has_repetition(text: str, threshold: int = 3) -> bool:
    """Check if text has repetitive patterns."""
    # Check for repeated sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2:
        return False

    # Count sentence occurrences
    counts = defaultdict(int)
    for s in sentences:
        normalized = re.sub(r'\s+', ' ', s.lower().strip())
        counts[normalized] += 1

    # Check if any sentence repeats too many times
    for count in counts.values():
        if count >= threshold:
            return True

    return False


def deduplicate_dataset(input_path: Path, output_path: Path, similarity_threshold: float = 0.85):
    """Deduplicate the dataset."""
    print(f"Loading dataset from: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Original samples: {len(data)}")

    # Track seen hashes and samples
    seen_hashes = set()
    seen_outputs = []  # Store outputs for similarity checking
    kept_samples = []

    stats = {
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "repetitive": 0,
        "kept": 0,
    }

    for i, sample in enumerate(data):
        if i % 5000 == 0:
            print(f"Processing {i}/{len(data)}...")

        # Get the combined text for deduplication
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        combined = f"{instruction} {input_text} {output}"
        sample_hash = compute_hash(combined)

        # Check exact duplicate
        if sample_hash in seen_hashes:
            stats["exact_duplicates"] += 1
            continue

        # Check for repetitive output
        if has_repetition(output, threshold=3):
            stats["repetitive"] += 1
            continue

        # Check near-duplicate (only against recent samples for efficiency)
        is_near_duplicate = False
        check_range = min(100, len(seen_outputs))  # Check last 100 outputs
        for prev_output in seen_outputs[-check_range:]:
            if similarity(output, prev_output) > similarity_threshold:
                is_near_duplicate = True
                stats["near_duplicates"] += 1
                break

        if is_near_duplicate:
            continue

        # Keep this sample
        seen_hashes.add(sample_hash)
        seen_outputs.append(output)
        kept_samples.append(sample)
        stats["kept"] += 1

    # Save deduplicated dataset
    print(f"\nSaving deduplicated dataset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(kept_samples, f, indent=2)

    # Print stats
    print("\n" + "=" * 50)
    print("Deduplication Statistics")
    print("=" * 50)
    print(f"Original samples:     {len(data):,}")
    print(f"Exact duplicates:     {stats['exact_duplicates']:,}")
    print(f"Near duplicates:      {stats['near_duplicates']:,}")
    print(f"Repetitive samples:   {stats['repetitive']:,}")
    print(f"Kept samples:         {stats['kept']:,}")
    print(f"Reduction:            {(1 - stats['kept']/len(data))*100:.1f}%")
    print("=" * 50)

    return kept_samples


if __name__ == "__main__":
    deduplicate_dataset(INPUT_PATH, OUTPUT_PATH)
