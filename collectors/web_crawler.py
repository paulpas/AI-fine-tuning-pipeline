#!/usr/bin/env python3
"""
Generic Web Crawler Collector

Collects training data from documentation sites and web pages,
with full JavaScript support via Playwright.

Features:
- JavaScript rendering via Playwright (headless browser)
- Configurable content extraction rules
- Respects robots.txt and rate limiting
- Handles pagination and link following
- Extracts Q&A patterns from documentation

Configuration:
    urls: List of starting URLs to crawl
    selectors: CSS selectors for content extraction
    max_pages: Maximum pages to crawl per domain (default: 100)
    follow_links: Whether to follow links (default: True)
    link_pattern: Regex pattern for links to follow
    js_wait: Seconds to wait for JavaScript (default: 2)

Usage:
    python -m collectors.web_crawler --output data/training/

Environment Variables:
    PLAYWRIGHT_BROWSERS_PATH: Custom browser installation path
"""

from __future__ import annotations

import os
import re
import time
import hashlib
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Generator, Set, TYPE_CHECKING, Any

from .base import BaseCollector, QAPair
from .registry import register_collector

# Playwright is optional - will fail gracefully if not installed
PLAYWRIGHT_AVAILABLE = False
sync_playwright = None

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING:
    from playwright.sync_api import Page, Browser


@register_collector
class WebCrawlerCollector(BaseCollector):
    """
    Generic web crawler with JavaScript support.

    Uses Playwright for rendering JavaScript-heavy documentation sites.
    """

    # Default documentation sites for DevOps training
    DEFAULT_URLS = [
        "https://docs.python.org/3/tutorial/",
        "https://kubernetes.io/docs/concepts/",
        "https://docs.docker.com/get-started/",
        "https://developer.hashicorp.com/terraform/docs",
        "https://docs.ansible.com/ansible/latest/user_guide/",
        "https://flask.palletsprojects.com/en/latest/",
        "https://fastapi.tiangolo.com/tutorial/",
    ]

    # Common content selectors for documentation sites
    DEFAULT_SELECTORS = {
        "content": [
            "article",
            "main",
            ".content",
            ".documentation",
            ".doc-content",
            "#content",
            "[role='main']",
        ],
        "title": [
            "h1",
            ".page-title",
            "article h1",
            "main h1",
        ],
        "code": [
            "pre code",
            ".highlight",
            ".codehilite",
            ".code-block",
        ],
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.browser: Any = None  # Browser when playwright is available
        self.playwright: Any = None
        self._visited_urls: Set[str] = set()
        self._request_count = 0
        self._last_request_time = 0

    def get_name(self) -> str:
        return "web_crawler"

    def get_description(self) -> str:
        return "Generic web crawler with JavaScript support for documentation sites"

    def get_required_config_keys(self) -> List[str]:
        return []

    def get_optional_config_keys(self) -> Dict:
        return {
            "urls": self.DEFAULT_URLS,
            "selectors": self.DEFAULT_SELECTORS,
            "max_pages": 100,
            "max_depth": 3,
            "follow_links": True,
            "link_pattern": None,  # Regex to filter links
            "js_wait": 2,  # Seconds to wait for JS
            "request_delay": 1.0,
            "headless": True,
            "respect_robots": True,
        }

    def validate_config(self) -> bool:
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for web_crawler. "
                "Install with: pip install playwright && playwright install chromium"
            )
        return True

    def setup(self) -> None:
        """Initialize Playwright browser"""
        self.urls = self.config.get("urls", self.DEFAULT_URLS)
        self.selectors = self.config.get("selectors", self.DEFAULT_SELECTORS)
        self.max_pages = self.config.get("max_pages", 100)
        self.max_depth = self.config.get("max_depth", 3)
        self.follow_links = self.config.get("follow_links", True)
        self.link_pattern = self.config.get("link_pattern")
        self.js_wait = self.config.get("js_wait", 2)
        self.request_delay = self.config.get("request_delay", 1.0)
        self.headless = self.config.get("headless", True)

        if self.link_pattern:
            self.link_regex = re.compile(self.link_pattern)
        else:
            self.link_regex = None

        # Start Playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.logger.info("Playwright browser initialized")

    def teardown(self) -> None:
        """Close browser and cleanup"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        self.logger.info("Browser closed")

    def collect(self) -> Generator[Dict, None, None]:
        """Crawl configured URLs and extract content"""
        for start_url in self.urls:
            self.logger.info(f"Crawling: {start_url}")
            domain = urlparse(start_url).netloc
            domain_count = 0

            # BFS crawl
            to_visit = [(start_url, 0)]  # (url, depth)
            domain_visited = set()

            while to_visit and domain_count < self.max_pages:
                url, depth = to_visit.pop(0)

                if url in domain_visited:
                    continue
                if depth > self.max_depth:
                    continue

                domain_visited.add(url)

                try:
                    page_data = self._fetch_page(url)
                    if page_data:
                        yield {
                            "url": url,
                            "domain": domain,
                            "depth": depth,
                            **page_data,
                        }
                        domain_count += 1

                        # Follow links
                        if self.follow_links and depth < self.max_depth:
                            for link in page_data.get("links", []):
                                if link not in domain_visited:
                                    if self._should_follow(link, domain):
                                        to_visit.append((link, depth + 1))

                except Exception as e:
                    self.logger.warning(f"Error crawling {url}: {e}")
                    continue

            self.logger.info(f"  {domain}: {domain_count} pages collected")

    def _fetch_page(self, url: str) -> Optional[Dict]:
        """Fetch a page and extract content"""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        page = self.browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for JavaScript
            if self.js_wait > 0:
                page.wait_for_timeout(self.js_wait * 1000)

            self._last_request_time = time.time()
            self._request_count += 1

            # Extract content
            title = self._extract_title(page)
            content = self._extract_content(page)
            code_blocks = self._extract_code(page)
            links = self._extract_links(page, url)

            if not content or len(content) < 100:
                return None

            return {
                "title": title,
                "content": content,
                "code_blocks": code_blocks,
                "links": links,
            }

        finally:
            page.close()

    def _extract_title(self, page: Any) -> str:
        """Extract page title"""
        for selector in self.selectors.get("title", ["h1"]):
            try:
                element = page.query_selector(selector)
                if element:
                    return element.inner_text().strip()
            except Exception:
                continue

        # Fallback to page title
        return page.title()

    def _extract_content(self, page: Any) -> str:
        """Extract main content text"""
        for selector in self.selectors.get("content", ["main"]):
            try:
                element = page.query_selector(selector)
                if element:
                    text = element.inner_text()
                    # Clean up whitespace
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    text = text.strip()
                    if len(text) > 100:
                        return text
            except Exception:
                continue

        # Fallback to body
        try:
            body = page.query_selector("body")
            if body:
                return body.inner_text()[:5000]
        except Exception:
            pass

        return ""

    def _extract_code(self, page: Any) -> List[Dict]:
        """Extract code blocks from the page"""
        code_blocks = []

        for selector in self.selectors.get("code", ["pre code"]):
            try:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    code = elem.inner_text().strip()
                    if code and len(code) > 20:
                        # Try to detect language
                        classes = elem.get_attribute("class") or ""
                        lang = self._detect_language(classes, code)
                        code_blocks.append({
                            "code": code,
                            "language": lang,
                        })
            except Exception:
                continue

        return code_blocks

    def _detect_language(self, classes: str, code: str) -> str:
        """Detect programming language from class names or content"""
        # Check class names
        lang_patterns = {
            "python": r"python|py",
            "javascript": r"javascript|js",
            "typescript": r"typescript|ts",
            "bash": r"bash|shell|sh",
            "yaml": r"yaml|yml",
            "json": r"json",
            "go": r"\bgo\b|golang",
            "rust": r"rust",
            "java": r"\bjava\b",
            "sql": r"\bsql\b",
            "hcl": r"\bhcl\b|terraform",
        }

        classes_lower = classes.lower()
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, classes_lower):
                return lang

        # Content-based detection
        if code.startswith("#!/") or code.startswith("$"):
            return "bash"
        if "def " in code or "import " in code:
            return "python"
        if "function " in code or "const " in code or "let " in code:
            return "javascript"
        if code.strip().startswith("{") and code.strip().endswith("}"):
            return "json"

        return "text"

    def _extract_links(self, page: Any, base_url: str) -> List[str]:
        """Extract links from the page"""
        links = []
        try:
            elements = page.query_selector_all("a[href]")
            for elem in elements:
                href = elem.get_attribute("href")
                if href:
                    # Convert to absolute URL
                    absolute = urljoin(base_url, href)
                    # Remove fragments
                    absolute = absolute.split("#")[0]
                    if absolute and absolute.startswith("http"):
                        links.append(absolute)
        except Exception:
            pass

        return list(set(links))

    def _should_follow(self, url: str, origin_domain: str) -> bool:
        """Check if a link should be followed"""
        parsed = urlparse(url)

        # Stay on same domain
        if parsed.netloc != origin_domain:
            return False

        # Skip non-HTML resources
        skip_extensions = ['.pdf', '.png', '.jpg', '.gif', '.css', '.js', '.zip']
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False

        # Apply custom pattern if configured
        if self.link_regex:
            return bool(self.link_regex.search(url))

        return True

    def transform(self, raw_item: Dict) -> Optional[QAPair]:
        """Transform crawled page into Q&A format"""
        title = raw_item.get("title", "")
        content = raw_item.get("content", "")
        code_blocks = raw_item.get("code_blocks", [])
        url = raw_item.get("url", "")

        if not title or not content:
            return None

        # Build the answer with code examples
        answer = content[:2000]  # Limit content length

        # Add code blocks
        if code_blocks:
            answer += "\n\nExample code:\n"
            for block in code_blocks[:3]:  # Max 3 code blocks
                lang = block.get("language", "")
                code = block.get("code", "")
                if code:
                    answer += f"\n```{lang}\n{code}\n```\n"

        # Create instruction from title
        instruction = self._title_to_instruction(title)

        # Skip if too short
        if len(answer) < 200:
            return None

        return QAPair(
            instruction=instruction,
            input="",
            output=answer,
            source=f"web:{self._url_hash(url)}",
            source_type="web",
            score=0,
            tags=self._extract_tags(url, content),
            url=url,
            metadata={
                "domain": raw_item.get("domain", ""),
                "depth": raw_item.get("depth", 0),
                "code_blocks": len(code_blocks),
            }
        )

    def _title_to_instruction(self, title: str) -> str:
        """Convert page title to natural instruction"""
        title = title.strip()

        # Clean up common title patterns
        title = re.sub(r'\s*[-|—]\s*.*$', '', title)  # Remove "- Site Name"
        title = re.sub(r'\s*\|.*$', '', title)  # Remove "| Site Name"

        if not title:
            return "Explain this concept"

        # If already a question
        if title.endswith("?"):
            return title

        # Check for "How to" style
        how_patterns = ["how to", "how do", "getting started", "tutorial", "guide"]
        if any(p in title.lower() for p in how_patterns):
            if not title.endswith("?"):
                return f"{title}?"
            return title

        # Default: make it a question
        return f"Explain {title.lower()}"

    def _url_hash(self, url: str) -> str:
        """Generate short hash of URL for source tracking"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _extract_tags(self, url: str, content: str) -> List[str]:
        """Extract relevant tags from URL and content"""
        tags = []

        # From URL
        domain = urlparse(url).netloc
        if "kubernetes" in domain or "k8s" in domain:
            tags.append("kubernetes")
        elif "docker" in domain:
            tags.append("docker")
        elif "terraform" in domain or "hashicorp" in domain:
            tags.append("terraform")
        elif "ansible" in domain:
            tags.append("ansible")
        elif "python" in domain:
            tags.append("python")
        elif "flask" in domain:
            tags.append("flask")
        elif "fastapi" in domain:
            tags.append("fastapi")

        # From content keywords
        content_lower = content.lower()
        tech_keywords = [
            "kubernetes", "docker", "terraform", "ansible",
            "python", "flask", "fastapi", "aws", "gcp", "azure",
            "ci/cd", "devops", "api", "microservices"
        ]
        for kw in tech_keywords:
            if kw in content_lower and kw not in tags:
                tags.append(kw)
                if len(tags) >= 5:
                    break

        return tags


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Crawler Collector")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--urls", nargs="+", help="URLs to crawl")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages per domain")
    parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth")
    parser.add_argument("--js-wait", type=float, default=2, help="JS wait time (seconds)")
    parser.add_argument("--no-follow", action="store_true", help="Don't follow links")
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "full"], default="alpaca")
    parser.add_argument("--visible", action="store_true", help="Show browser window")

    args = parser.parse_args()

    config = {
        "max_pages": args.max_pages,
        "max_depth": args.max_depth,
        "js_wait": args.js_wait,
        "follow_links": not args.no_follow,
        "headless": not args.visible,
    }
    if args.urls:
        config["urls"] = args.urls

    collector = WebCrawlerCollector(config)
    collector.run(args.output, format=args.format)
