import time
import hashlib
import json
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from pathlib import Path
from typing import Set, List, Tuple, Optional
import sys

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Browser, BrowserContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    START_URL,
    ALLOWED_DOMAIN,
    MAX_DEPTH,
    REQUEST_DELAY,
    TIMEOUT,
    USER_AGENT,
    RAW_HTML_DIR,
    CRAWLER_STATE_FILE,
    STATE_SAVE_INTERVAL,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Stats tracking
class CrawlStats:
    def __init__(self):
        self.start_time = datetime.now()
        self.pages_crawled = 0
        self.pages_failed = 0
        self.bytes_downloaded = 0
        self.last_log_time = datetime.now()
        self.log_interval = 10  # Log summary every N pages

    def log_progress(self, queue_size: int, current_url: str, depth: int):
        self.pages_crawled += 1
        elapsed = datetime.now() - self.start_time
        rate = self.pages_crawled / max(elapsed.total_seconds() / 60, 0.1)

        # Always log current URL
        log.info(f"[{self.pages_crawled}] CRAWLING: {current_url}")
        log.info(f"    Depth: {depth}/{MAX_DEPTH} | Queue: {queue_size} | Rate: {rate:.1f} pages/min")

        # Log summary every N pages
        if self.pages_crawled % self.log_interval == 0:
            self._log_summary(queue_size)

    def _log_summary(self, queue_size: int):
        elapsed = datetime.now() - self.start_time
        log.info("=" * 60)
        log.info(f"PROGRESS SUMMARY")
        log.info(f"  Pages crawled: {self.pages_crawled}")
        log.info(f"  Pages failed:  {self.pages_failed}")
        log.info(f"  Queue size:    {queue_size}")
        log.info(f"  Data size:     {self.bytes_downloaded / 1024 / 1024:.1f} MB")
        log.info(f"  Elapsed:       {str(elapsed).split('.')[0]}")
        log.info("=" * 60)


class HashiCorpCrawler:
    def __init__(self, resume: bool = True) -> None:
        self.visited: Set[str] = set()
        self.to_visit: List[Tuple[str, int]] = [(START_URL, 0)]
        self.rp = robotparser.RobotFileParser()
        self.rp.set_url(urljoin(START_URL, "/robots.txt"))
        self.rp.read()
        self.stats = CrawlStats()
        self.resume_mode = resume

        # Try to resume from saved state or existing files
        if resume:
            self._try_resume()

        log.info(f"Crawler initialized - Start URL: {START_URL}")
        log.info(f"Settings: MAX_DEPTH={MAX_DEPTH}, DELAY={REQUEST_DELAY}s, TIMEOUT={TIMEOUT}s")
        log.info(f"Resume mode: {resume} | Visited: {len(self.visited)} | Queue: {len(self.to_visit)}")

    def _try_resume(self) -> None:
        """Try to resume from saved state or rebuild from existing files."""
        # First try to load saved state
        if self._load_state():
            return

        # Otherwise rebuild visited set from existing HTML files
        self._rebuild_from_files()

    def _load_state(self) -> bool:
        """Load crawler state from disk. Returns True if successful."""
        if not CRAWLER_STATE_FILE.exists():
            log.info("No saved state found")
            return False

        try:
            state = json.loads(CRAWLER_STATE_FILE.read_text(encoding="utf-8"))
            self.visited = set(state.get("visited", []))
            self.to_visit = [tuple(x) for x in state.get("to_visit", [])]
            self.stats.pages_crawled = state.get("pages_crawled", 0)
            self.stats.pages_failed = state.get("pages_failed", 0)
            self.stats.bytes_downloaded = state.get("bytes_downloaded", 0)

            log.info("=" * 60)
            log.info("RESUMING FROM SAVED STATE")
            log.info(f"  Visited URLs: {len(self.visited)}")
            log.info(f"  Queue size: {len(self.to_visit)}")
            log.info(f"  Pages crawled: {self.stats.pages_crawled}")
            log.info("=" * 60)
            return True
        except Exception as e:
            log.warning(f"Failed to load state: {e}")
            return False

    def _rebuild_from_files(self) -> None:
        """Rebuild visited set from existing HTML files and their JSON metadata."""
        if not RAW_HTML_DIR.exists():
            log.info("No existing files to rebuild from")
            return

        json_files = list(RAW_HTML_DIR.glob("*.json"))
        if not json_files:
            log.info("No existing files to rebuild from")
            return

        log.info(f"Rebuilding state from {len(json_files)} existing files...")

        for json_path in json_files:
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                url = meta.get("url")
                if url:
                    norm = self._normalise(url)
                    self.visited.add(norm)
            except Exception as e:
                log.debug(f"Failed to read {json_path}: {e}")

        # Calculate approximate bytes from HTML files
        for html_path in RAW_HTML_DIR.glob("*.html"):
            self.stats.bytes_downloaded += html_path.stat().st_size

        self.stats.pages_crawled = len(self.visited)

        log.info("=" * 60)
        log.info("REBUILT STATE FROM EXISTING FILES")
        log.info(f"  Already crawled: {len(self.visited)} pages")
        log.info(f"  Data size: {self.stats.bytes_downloaded / 1024 / 1024:.1f} MB")
        log.info(f"  Queue size: {len(self.to_visit)}")
        log.info("=" * 60)

    def _save_state(self) -> None:
        """Save current crawler state to disk."""
        state = {
            "visited": list(self.visited),
            "to_visit": self.to_visit,
            "pages_crawled": self.stats.pages_crawled,
            "pages_failed": self.stats.pages_failed,
            "bytes_downloaded": self.stats.bytes_downloaded,
            "saved_at": datetime.now().isoformat(),
        }
        CRAWLER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CRAWLER_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        log.debug(f"State saved: {len(self.visited)} visited, {len(self.to_visit)} queued")

    def clear_state(self) -> None:
        """Clear saved state to start fresh."""
        if CRAWLER_STATE_FILE.exists():
            CRAWLER_STATE_FILE.unlink()
            log.info("Cleared saved crawler state")

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

        # Track bytes downloaded
        html_bytes = len(html.encode('utf-8'))
        self.stats.bytes_downloaded += html_bytes

        # also store a tiny JSON manifest for later stages
        title = self._extract_title(html)
        meta = {
            "url": url,
            "title": title,
            "filename": str(path.relative_to(RAW_HTML_DIR)),
        }
        (RAW_HTML_DIR / f"{h}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"    SAVED: {title[:60]}... ({html_bytes / 1024:.1f} KB)")

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
        log.info("=" * 60)
        log.info("STARTING CRAWL")
        log.info("=" * 60)

        pages_this_session = 0

        with sync_playwright() as p:
            log.info("Launching headless browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT)
            log.info("Browser ready")

            try:
                while self.to_visit:
                    url, depth = self.to_visit.pop(0)
                    if depth > MAX_DEPTH:
                        log.debug(f"Skipping (depth {depth} > {MAX_DEPTH}): {url}")
                        continue
                    norm = self._normalise(url)
                    if norm in self.visited:
                        continue
                    self.visited.add(norm)

                    # Log progress before fetching
                    self.stats.log_progress(len(self.to_visit), url, depth)

                    try:
                        page = context.new_page()
                        page.goto(url, wait_until="networkidle", timeout=TIMEOUT * 1000)
                        html = page.content()
                        page.close()

                        self._store_page(url, html)
                        links_before = len(self.to_visit)
                        self._enqueue_links(html, url, depth)
                        links_added = len(self.to_visit) - links_before
                        if links_added > 0:
                            log.info(f"    LINKS: +{links_added} new URLs added to queue")

                        pages_this_session += 1

                    except Exception as exc:
                        self.stats.pages_failed += 1
                        log.warning(f"    FAILED: {exc}")

                    # Save state periodically
                    if pages_this_session % STATE_SAVE_INTERVAL == 0:
                        self._save_state()
                        log.info(f"    STATE SAVED (checkpoint)")

                    time.sleep(REQUEST_DELAY)

            except KeyboardInterrupt:
                log.warning("Interrupted by user - saving state before exit...")
                self._save_state()
                raise
            finally:
                # Always save state on exit
                self._save_state()
                context.close()
                browser.close()

        # Clear state file on successful completion
        if not self.to_visit:
            self.clear_state()
            log.info("Crawl completed - state file cleared")

        # Final summary
        log.info("=" * 60)
        log.info("CRAWL COMPLETE")
        log.info(f"  Total pages crawled: {self.stats.pages_crawled}")
        log.info(f"  Pages this session:  {pages_this_session}")
        log.info(f"  Total pages failed:  {self.stats.pages_failed}")
        log.info(f"  Total data:          {self.stats.bytes_downloaded / 1024 / 1024:.1f} MB")
        elapsed = datetime.now() - self.stats.start_time
        log.info(f"  Session time:        {str(elapsed).split('.')[0]}")
        log.info("=" * 60)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    crawler = HashiCorpCrawler()
    crawler.crawl()
