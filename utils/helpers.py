import hashlib
import uuid
from pathlib import Path
from typing import Any

def token_count(text: str) -> int:
    """
    Rough token estimator – 1 token ≈ 4 characters for English text.
    Replace with a proper tokenizer (e.g., tiktoken) for production.
    """
    return max(1, len(text) // 4)


def deterministic_uuid(*parts: Any) -> str:
    """
    Produce a stable UUID5 (SHA‑1 namespace) from the concatenated parts.
    Guarantees the same chunk_id across runs.
    """
    ns = uuid.NAMESPACE_URL
    name = "-".join(map(str, parts))
    return str(uuid.uuid5(ns, name))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
