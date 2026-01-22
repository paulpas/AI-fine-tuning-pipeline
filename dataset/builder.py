import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, FINAL_DATASET
from utils.helpers import sha256_file

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _load_supervision() -> List[Dict]:
    path = DATA_DIR / "supervision.jsonl"
    if not path.exists():
        raise FileNotFoundError("Supervision file not found – run synthetic generator first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _filter_grounded(samples: List[Dict]) -> List[Dict]:
    """
    Very lightweight filter:
      – drop any sample whose output contains the placeholder token "<...>"
      – drop if output length < 5 words (likely empty)
    Real‑world pipelines would run an entailment model or regex checks.
    """
    filtered = []
    for s in samples:
        out = s["output"]
        if "<" in out and ">" in out:
            continue
        if len(out.split()) < 5:
            continue
        filtered.append(s)
    return filtered


def assemble_dataset() -> None:
    """Create final JSONL + metadata file."""
    samples = _load_supervision()
    log.info("Loaded %d raw supervision samples", len(samples))

    filtered = _filter_grounded(samples)
    log.info("After grounding filter → %d samples", len(filtered))

    # Deterministic ordering by hash of instruction+output
    filtered.sort(key=lambda x: hashlib.sha256((x["instruction"] + x["output"]).encode()).hexdigest())

    # Write JSONL
    with FINAL_DATASET.open("w", encoding="utf-8") as f:
        for s in filtered:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write small metadata JSON alongside
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_urls": list({s["_meta"]["source_url"] for s in filtered}),
        "raw_content_hash": sha256_file(DATA_DIR / "supervision.jsonl"),
        "num_samples": len(filtered),
    }
    (FINAL_DATASET.parent / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Final dataset written to %s (metadata in %s)",
             FINAL_DATASET, FINAL_DATASET.parent / "dataset_meta.json")


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    assemble_dataset()
