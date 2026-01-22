import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict
import sys

from bs4 import BeautifulSoup
from readability import Document
import markdownify

# Silence noisy INFO messages from readability (not actual errors)
logging.getLogger("readability.readability").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_HTML_DIR, MARKDOWN_DIR

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

    html_files = _load_raw_pages()
    total = len(html_files)
    log.info(f"Found {total} HTML files to extract")

    extracted = 0
    failed = 0
    total_md_size = 0

    for i, html_path in enumerate(html_files, 1):
        try:
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

            md_size = len(md)
            total_md_size += md_size
            extracted += 1

            log.info(f"[{i}/{total}] EXTRACTED: {meta['title'][:50]}...")
            log.info(f"    URL: {meta['url']}")
            log.info(f"    Markdown: {md_size / 1024:.1f} KB, {len(_extract_headings(md))} headings")

        except Exception as e:
            failed += 1
            log.warning(f"[{i}/{total}] FAILED: {html_path.name} - {e}")

        # Progress summary every 100 files
        if i % 100 == 0:
            log.info("=" * 50)
            log.info(f"EXTRACTION PROGRESS: {i}/{total} ({i*100//total}%)")
            log.info(f"  Extracted: {extracted}, Failed: {failed}")
            log.info(f"  Total markdown: {total_md_size / 1024 / 1024:.1f} MB")
            log.info("=" * 50)

    log.info("=" * 50)
    log.info("EXTRACTION COMPLETE")
    log.info(f"  Files processed: {total}")
    log.info(f"  Extracted: {extracted}")
    log.info(f"  Failed: {failed}")
    log.info(f"  Total markdown: {total_md_size / 1024 / 1024:.1f} MB")
    log.info("=" * 50)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    extract_all()
