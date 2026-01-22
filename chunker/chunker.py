import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MARKDOWN_DIR, CHUNKS_DIR
from utils.helpers import token_count, deterministic_uuid

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

    docs = _load_documents()
    total_docs = len(docs)
    log.info(f"Found {total_docs} documents to chunk")
    log.info(f"Settings: MIN_TOKENS={MIN_TOKENS}, MAX_TOKENS={MAX_TOKENS}")

    total_chunks = 0
    total_sections = 0

    for i, doc in enumerate(docs, 1):
        sections = _split_by_headings(doc["markdown"], doc["headings"])
        all_chunks: List[Dict[str, Any]] = []
        for sec in sections:
            all_chunks.extend(_chunk_section(sec, doc))

        # Write chunks for this document
        for ch in all_chunks:
            out_path = CHUNKS_DIR / f"{ch['chunk_id']}.json"
            out_path.write_text(json.dumps(ch, ensure_ascii=False, indent=2), encoding="utf-8")

        total_chunks += len(all_chunks)
        total_sections += len(sections)

        log.info(f"[{i}/{total_docs}] CHUNKED: {doc['title'][:50]}...")
        log.info(f"    Sections: {len(sections)}, Chunks: {len(all_chunks)}")

        # Progress summary every 100 docs
        if i % 100 == 0:
            log.info("=" * 50)
            log.info(f"CHUNKING PROGRESS: {i}/{total_docs} ({i*100//total_docs}%)")
            log.info(f"  Total sections: {total_sections}")
            log.info(f"  Total chunks: {total_chunks}")
            log.info("=" * 50)

    log.info("=" * 50)
    log.info("CHUNKING COMPLETE")
    log.info(f"  Documents processed: {total_docs}")
    log.info(f"  Total sections: {total_sections}")
    log.info(f"  Total chunks: {total_chunks}")
    log.info(f"  Avg chunks/doc: {total_chunks / max(total_docs, 1):.1f}")
    log.info("=" * 50)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    create_chunks()
