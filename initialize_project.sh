#!/usr/bin/env bash
# ----------------------------------------------------------------------
# setup_pipeline.sh
# One‑shot script to materialise the full prototype pipeline described
# in the previous answer.
#
#   • Run from inside the already‑created folder `hashicorp_terraform_dataset`
#   • After it finishes you will have:
#        data/
#        hashicorp_terraform_dataset/   (the Python package)
#        run_pipeline.py
#        requirements.txt
#
#   • The pipeline will stop at the synthetic‑supervision stage because
#     the LLM call is still a stub.  Replace the stub in
#     synthetic/generator.py with a real API call and re‑run `python run_pipeline.py`.
# ----------------------------------------------------------------------

set -euo pipefail

# --------------------------------------------------------------
# Helper: write a file only if the content differs (idempotent)
# --------------------------------------------------------------
write_file() {
    local target_path="$1"
    local tmp_file
    tmp_file=$(mktemp)

    cat >"$tmp_file"
    if ! cmp -s "$target_path" "$tmp_file" 2>/dev/null; then
        mkdir -p "$(dirname "$target_path")"
        mv "$tmp_file" "$target_path"
        echo "✔  Created/updated $target_path"
    else
        rm "$tmp_file"
        echo "✔  $target_path already up‑to‑date"
    fi
}

# --------------------------------------------------------------
# 1️⃣  Directory skeleton (including empty __init__.py files)
# --------------------------------------------------------------
for pkg in crawler extractor chunker synthetic dataset utils; do
    mkdir -p "$pkg"
    write_file "$pkg/__init__.py" <<'PY_EOF'
# This file intentionally left blank – marks the directory as a Python package.
PY_EOF
done

# --------------------------------------------------------------
# 2️⃣  Core Python modules
# --------------------------------------------------------------

# ── config.py
write_file "config.py" <<'PY_EOF'
import os
from pathlib import Path

# --------------------------------------------------------------
# Core URLs
# --------------------------------------------------------------
START_URL = "https://developer.hashicorp.com/tutorials/library?product=terraform"
ALLOWED_DOMAIN = "developer.hashicorp.com"

# --------------------------------------------------------------
# Crawl limits
# --------------------------------------------------------------
MAX_DEPTH = 5               # safety guard against runaway recursion
REQUEST_DELAY = 1.0         # seconds between HTTP requests
TIMEOUT = 15                # request timeout in seconds

# --------------------------------------------------------------
# Storage locations (relative to project root)
# --------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"

RAW_HTML_DIR = DATA_DIR / "raw_html"
MARKDOWN_DIR = DATA_DIR / "markdown"
CHUNKS_DIR = DATA_DIR / "chunks"
FINAL_DATASET = DATA_DIR / "dataset.jsonl"

# --------------------------------------------------------------
# Misc
# --------------------------------------------------------------
USER_AGENT = "HashiCorpDatasetBot/0.1 (+https://github.com/yourorg)"
RANDOM_SEED = 42
PY_EOF

# ── utils/helpers.py
write_file "utils/helpers.py" <<'PY_EOF'
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
PY_EOF

# ── crawler/crawler.py
write_file "crawler/crawler.py" <<'PY_EOF'
import time
import hashlib
import json
import logging
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from pathlib import Path
from typing import Set, List, Tuple

import requests
from bs4 import BeautifulSoup

from ..config import (
    START_URL,
    ALLOWED_DOMAIN,
    MAX_DEPTH,
    REQUEST_DELAY,
    TIMEOUT,
    USER_AGENT,
    RAW_HTML_DIR,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class HashiCorpCrawler:
    def __init__(self) -> None:
        self.visited: Set[str] = set()
        self.to_visit: List[Tuple[str, int]] = [(START_URL, 0)]
        self.rp = robotparser.RobotFileParser()
        self.rp.set_url(urljoin(START_URL, "/robots.txt"))
        self.rp.read()

    @staticmethod
    def _normalise(url: str) -> str:
        """Strip fragments, sort query parameters and remove trailing slash."""
        parsed = urlparse(url)
        cleaned = parsed._replace(fragment="").geturl()
        if cleaned.endswith("/"):
            cleaned = cleaned[:-1]
        return cleaned

    def _allowed(self, url: str) -> bool:
        """Check domain, robots.txt and that we have not visited before."""
        parsed = urlparse(url)
        if parsed.netloc != ALLOWED_DOMAIN:
            return False
        norm = self._normalise(url)
        if norm in self.visited:
            return False
        if not self.rp.can_fetch(USER_AGENT, url):
            log.debug("Disallowed by robots.txt: %s", url)
            return False
        return True

    def _store_page(self, url: str, html: str) -> None:
        """Write raw HTML to disk, using a deterministic filename."""
        h = hashlib.sha256(url.encode()).hexdigest()
        path = RAW_HTML_DIR / f"{h}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        # also store a tiny JSON manifest for later stages
        meta = {
            "url": url,
            "title": self._extract_title(html),
            "filename": str(path.relative_to(RAW_HTML_DIR)),
        }
        (RAW_HTML_DIR / f"{h}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Saved %s (%d bytes)", url, len(html))

    @staticmethod
    def _extract_title(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        if title_tag := soup.find("title"):
            return title_tag.get_text(strip=True)
        return ""

    def _enqueue_links(self, html: str, base_url: str, depth: int) -> None:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Resolve relative URLs
            next_url = urljoin(base_url, href)
            if self._allowed(next_url):
                self.to_visit.append((next_url, depth + 1))

    def crawl(self) -> None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        while self.to_visit:
            url, depth = self.to_visit.pop(0)
            if depth > MAX_DEPTH:
                continue
            norm = self._normalise(url)
            if norm in self.visited:
                continue
            self.visited.add(norm)

            try:
                resp = session.get(url, timeout=TIMEOUT)
                resp.raise_for_status()
                html = resp.text
                self._store_page(url, html)
                self._enqueue_links(html, url, depth)
            except requests.RequestException as exc:
                log.warning("Failed %s: %s", url, exc)

            time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    crawler = HashiCorpCrawler()
    crawler.crawl()
PY_EOF

# ── extractor/extractor.py
write_file "extractor/extractor.py" <<'PY_EOF'
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

from bs4 import BeautifulSoup
from readability import Document
import markdownify

from ..config import RAW_HTML_DIR, MARKDOWN_DIR

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _load_raw_pages() -> List[Path]:
    """Return list of HTML files under RAW_HTML_DIR."""
    return sorted(RAW_HTML_DIR.rglob("*.html"))


def _read_html(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_main_content(html: str) -> str:
    """Use readability‑lxml to pull the article body."""
    doc = Document(html)
    return doc.summary()


def _html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown while preserving:
      * code blocks (`<pre><code>`)
      * inline code (`<code>`)
      * headings hierarchy
      * tables (as markdown tables)
    """
    return markdownify.markdownify(
        html,
        heading_style="ATX",
        code_language="preserve",
        bullet="*",
        strip=["script", "style"],
    )


def _slugify(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _extract_headings(md: str) -> List[str]:
    """Simple heading extractor – lines that start with '#'. Returns full heading text."""
    headings = []
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("#"):
            headings.append(line.lstrip("#").strip())
    return headings


def extract_all() -> None:
    """Read raw HTML, convert to clean markdown, and write a JSONL manifest."""
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    for html_path in _load_raw_pages():
        html = _read_html(html_path)
        main_html = _extract_main_content(html)
        md = _html_to_markdown(main_html)

        # Load the side‑car JSON to get URL & title
        meta_path = html_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        doc_id = _slugify(meta["url"])
        out = {
            "doc_id": doc_id,
            "url": meta["url"],
            "title": meta["title"],
            "headings": _extract_headings(md),
            "markdown": md,
        }

        out_path = MARKDOWN_DIR / f"{doc_id}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Extracted %s → %s", meta["url"], out_path.name)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    extract_all()
PY_EOF

# ── chunker/chunker.py
write_file "chunker/chunker.py" <<'PY_EOF'
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

from ..config import MARKDOWN_DIR, CHUNKS_DIR
from ..utils.helpers import token_count, deterministic_uuid

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# --------------------------------------------------------------
# Settings (can be moved to config later)
# --------------------------------------------------------------
MIN_TOKENS = 512
MAX_TOKENS = 1024


def _load_documents() -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(MARKDOWN_DIR.glob("*.json"))]


def _split_by_headings(md: str, headings: List[str]) -> List[Dict[str, str]]:
    """
    Very naive splitter: split the markdown at heading lines.
    Returns a list of sections with keys:
        - "section_title"
        - "content"
    """
    sections: List[Dict[str, str]] = []
    current_title = "intro"
    current_lines: List[str] = []

    for line in md.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # store previous section
            if current_lines:
                sections.append({"section_title": current_title, "content": "\n".join(current_lines)})
                current_lines = []
            # new heading
            current_title = stripped.lstrip("#").strip()
        else:
            current_lines.append(line)

    # final section
    if current_lines:
        sections.append({"section_title": current_title, "content": "\n".join(current_lines)})
    return sections


def _chunk_section(section: Dict[str, str], doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Break a section into token‑bounded chunks.
    We never split code fences (``` … ```) or markdown tables.
    """
    chunks: List[Dict[str, Any]] = []
    text = section["content"]
    lines = text.splitlines()
    buffer: List[str] = []
    inside_fence = False
    inside_table = False

    for line in lines:
        # Detect start/end of fenced code block
        if line.strip().startswith("```"):
            inside_fence = not inside_fence

        # Detect simple markdown tables (line containing '|')
        if not inside_fence and "|" in line and line.strip().startswith("|"):
            inside_table = True
        elif inside_table and line.strip() == "":
            inside_table = False

        buffer.append(line)

        # Only consider breaking when we are outside protected blocks
        if not inside_fence and not inside_table:
            cur_tokens = token_count("\n".join(buffer))
            if cur_tokens >= MAX_TOKENS:
                # Emit a chunk
                chunk_text = "\n".join(buffer)
                chunk_id = deterministic_uuid(doc_meta["doc_id"], section["section_title"], len(chunks))
                chunks.append(
                    {
                        "doc_id": doc_meta["doc_id"],
                        "url": doc_meta["url"],
                        "title": doc_meta["title"],
                        "section": section["section_title"],
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )
                buffer = []

    # Flush remaining buffer (ensure minimum size)
    if buffer:
        chunk_text = "\n".join(buffer)
        if token_count(chunk_text) >= MIN_TOKENS or len(chunks) == 0:
            chunk_id = deterministic_uuid(doc_meta["doc_id"], section["section_title"], len(chunks))
            chunks.append(
                {
                    "doc_id": doc_meta["doc_id"],
                    "url": doc_meta["url"],
                    "title": doc_meta["title"],
                    "section": section["section_title"],
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )
    return chunks


def create_chunks() -> None:
    """Read markdown docs → split → write each chunk as JSON."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for doc in _load_documents():
        sections = _split_by_headings(doc["markdown"], doc["headings"])
        all_chunks: List[Dict[str, Any]] = []
        for sec in sections:
            all_chunks.extend(_chunk_section(sec, doc))

        # Write chunks for this document
        for ch in all_chunks:
            out_path = CHUNKS_DIR / f"{ch['chunk_id']}.json"
            out_path.write_text(json.dumps(ch, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Created %d chunks for %s", len(all_chunks), doc["url"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    create_chunks()
PY_EOF

# ── synthetic/generator.py
write_file "synthetic/generator.py" <<'PY_EOF'
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from ..config import CHUNKS_DIR, DATA_DIR
from ..utils.helpers import deterministic_uuid

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# --------------------------------------------------------------
# Prompt template (LLM‑side)
# --------------------------------------------------------------
PROMPT_TEMPLATE = """
You are given a technical excerpt from HashiCorp Terraform documentation.

Excerpt:
\"\"\"
{chunk}
\"\"\"

Based **only** on the excerpt, produce up to 5 distinct instruction‑answer pairs.
Each pair must be one of:
  • factual Q&A,
  • "Explain how this works",
  • "How do I perform X",
  • a failure scenario with a resolution,
  • an architectural explanation (if applicable).

If the excerpt does not contain enough information for a type, simply omit it.
Return a JSON array where each element has the keys:
  "instruction", "input" (always empty), "output".
Do NOT hallucinate any facts outside the provided text.
"""

# --------------------------------------------------------------
# Stub – replace with real LLM call (e.g., OpenAI, Anthropic, etc.)
# --------------------------------------------------------------
def _call_llm(prompt: str) -> List[Dict[str, str]]:
    """
    TODO: integrate with your LLM provider.
    For now we raise NotImplementedError so the pipeline can be run up to
    this point without external API access.
    """
    raise NotImplementedError("LLM integration not implemented yet.")


def generate_supervision() -> None:
    """Iterate over chunk files, ask the LLM for supervision, and write JSONL."""
    out_path = DATA_DIR / "supervision.jsonl"
    with out_path.open("w", encoding="utf-8") as writer:
        for chunk_file in sorted(CHUNKS_DIR.glob("*.json")):
            chunk = json.loads(chunk_file.read_text(encoding="utf-8"))
            prompt = PROMPT_TEMPLATE.format(chunk=chunk["text"])
            try:
                samples = _call_llm(prompt)
            except NotImplementedError:
                log.warning("LLM stub hit – stop processing further chunks.")
                break

            for s in samples:
                # Attach minimal provenance (optional)
                s["_meta"] = {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "source_url": chunk["url"],
                }
                writer.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("Supervision data written to %s", out_path)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    generate_supervision()
PY_EOF

# ── dataset/builder.py
write_file "dataset/builder.py" <<'PY_EOF'
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from ..config import DATA_DIR, FINAL_DATASET
from ..utils.helpers import sha256_file

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
PY_EOF

# ── run_pipeline.py (top‑level orchestrator)
write_file "run_pipeline.py" <<'PY_EOF'
import logging
from pathlib import Path

from crawler.crawler import HashiCorpCrawler
from extractor.extractor import extract_all
from chunker.chunker import create_chunks
from synthetic.generator import generate_supervision
from dataset.builder import assemble_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s", level=logging.INFO
)

def main() -> None:
    # 1️⃣ Crawl
    logging.info("=== CRAWLING START ===")
    crawler = HashiCorpCrawler()
    crawler.crawl()
    logging.info("=== CRAWLING DONE ===")

    # 2️⃣ Extract → markdown
    logging.info("=== EXTRACTION START ===")
    extract_all()
    logging.info("=== EXTRACTION DONE ===")

    # 3️⃣ Chunk
    logging.info("=== CHUNKING START ===")
    create_chunks()
    logging.info("=== CHUNKING DONE ===")

    # 4️⃣ Synthetic supervision (LLM stub – will raise NotImplementedError)
    logging.info("=== SYNTHETIC SUPERVISION START ===")
    try:
        generate_supervision()
    except NotImplementedError as e:
        logging.warning("LLM integration not yet wired – stopping pipeline here.")
        return
    logging.info("=== SYNTHETIC SUPERVISION DONE ===")

    # 5️⃣ Final dataset assembly
    logging.info("=== DATASET BUILD START ===")
    assemble_dataset()
    logging.info("=== PIPELINE COMPLETE ===")
    logging.info(f"Dataset ready at: {Path('hashicorp_terraform_dataset/data/dataset.jsonl')}")

if __name__ == "__main__":
    main()
PY_EOF

# --------------------------------------------------------------
# 4️⃣ requirements.txt
# --------------------------------------------------------------
write_file "requirements.txt" <<'REQ_EOF'
beautifulsoup4==4.12.3
requests==2.32.3
readability-lxml==0.8.1
markdownify==0.11.6
tiktoken==0.7.0   # optional – replace token_count with a real tokenizer later
REQ_EOF

# --------------------------------------------------------------
# 5️⃣ (optional) create a virtual‑env and install deps
# --------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating isolated virtual‑env in .venv using uv ..."
    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        echo "Installing uv (fast Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv
fi

# Activate the venv for the rest of the script
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing required Python packages with uv ..."
uv pip install -r requirements.txt

# --------------------------------------------------------------
# 6️⃣ Run the pipeline (it will stop after the LLM stub)
# --------------------------------------------------------------
echo "Running the pipeline (will stop at the synthetic‑supervision stage)..."
python run_pipeline.py || true   # we intentionally ignore the exit‑code from the stub

# --------------------------------------------------------------
# 7️⃣ Final note to the user
# --------------------------------------------------------------
cat <<'END_MSG'

✅  All files have been created.
✅  A virtual‑environment `.venv` has been set up and dependencies installed.
✅  The pipeline has been executed up to the point where a real LLM call is required.

🛠️  Next steps:
   1. Open `synthetic/generator.py` and replace the `_call_llm` function
      with a call to your preferred LLM API (OpenAI, Anthropic, Cohere, …).
   2. Re‑run the pipeline:
        source .venv/bin/activate
        python run_pipeline.py
   3. When the run finishes you will find the final JSONL at:
        data/dataset.jsonl
   4. The accompanying metadata file is `data/dataset_meta.json`.

🚨  Remember to respect HashiCorp’s usage policies and robots.txt
      when you actually hit their LLM endpoints.

Happy hacking!
END_MSG
