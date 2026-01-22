#!/usr/bin/env python3
"""
Stack Exchange Collector

Collects high-quality Q&A from Stack Overflow and other Stack Exchange sites
via the official API.

Features:
- Fetches questions by tag with configurable filters
- Gets accepted or highest-voted answers
- Converts HTML to clean markdown
- Respects API rate limits

Configuration:
    api_key: Stack Exchange API key (optional but recommended)
    tags: List of tags to collect (e.g., ["python", "kubernetes"])
    min_score: Minimum question score (default: 10)
    max_per_tag: Maximum questions per tag (default: 200)
    site: Stack Exchange site (default: "stackoverflow")

Usage:
    python -m collectors.stackoverflow --output data/training/

Environment Variables:
    STACKOVERFLOW_API_KEY: API key for higher rate limits
"""

import os
import re
import html
import time
from typing import Dict, List, Optional, Generator

import requests

from .base import BaseCollector, QAPair
from .registry import register_collector


@register_collector
class StackExchangeCollector(BaseCollector):
    """
    Collector for Stack Exchange sites (Stack Overflow, Server Fault, etc.)
    """

    BASE_URL = "https://api.stackexchange.com/2.3"

    # Default tags for DevOps-focused collection
    DEFAULT_TAGS = [
        "python",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "aws",
        "boto3",
        "pytest",
        "flask",
        "fastapi",
        "asyncio",
        "github-actions",
    ]

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.session = requests.Session()
        self._request_count = 0
        self._last_request_time = 0

    def get_name(self) -> str:
        return "stackoverflow"

    def get_description(self) -> str:
        return "Collects Q&A from Stack Overflow via the Stack Exchange API"

    def get_required_config_keys(self) -> List[str]:
        return []  # API key is optional

    def get_optional_config_keys(self) -> Dict:
        return {
            "api_key": None,
            "tags": self.DEFAULT_TAGS,
            "min_score": 10,
            "max_per_tag": 200,
            "site": "stackoverflow",
            "request_delay": 0.5,
        }

    def setup(self) -> None:
        """Initialize API key from config or environment"""
        self.api_key = self.config.get("api_key") or os.environ.get("STACKOVERFLOW_API_KEY")
        self.tags = self.config.get("tags", self.DEFAULT_TAGS)
        self.min_score = self.config.get("min_score", 10)
        self.max_per_tag = self.config.get("max_per_tag", 200)
        self.site = self.config.get("site", "stackoverflow")
        self.request_delay = self.config.get("request_delay", 0.5)

        if self.api_key:
            self.logger.info("Using Stack Exchange API key")
        else:
            self.logger.warning("No API key - rate limits will be strict")

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        params["site"] = self.site
        params["filter"] = "withbody"

        if self.api_key:
            params["key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params)

        self._last_request_time = time.time()
        self._request_count += 1

        # Handle rate limiting
        if response.status_code == 429:
            backoff = int(response.headers.get("Retry-After", 60))
            self.logger.warning(f"Rate limited. Waiting {backoff}s...")
            time.sleep(backoff)
            return self._make_request(endpoint, params)

        response.raise_for_status()
        return response.json()

    def collect(self) -> Generator[Dict, None, None]:
        """Collect questions and answers for configured tags"""
        for tag in self.tags:
            self.logger.info(f"Collecting questions for tag: {tag}")
            count = 0

            for question in self._get_questions(tag):
                if count >= self.max_per_tag:
                    break

                # Get the best answer
                answer = self._get_best_answer(question["question_id"])
                if answer:
                    yield {
                        "question": question,
                        "answer": answer,
                        "tag": tag,
                    }
                    count += 1

            self.logger.info(f"  {tag}: {count} Q&A pairs collected")

    def _get_questions(self, tag: str) -> Generator[Dict, None, None]:
        """Get top questions for a tag"""
        page = 1
        max_pages = 5

        while page <= max_pages:
            params = {
                "tagged": tag,
                "sort": "votes",
                "order": "desc",
                "pagesize": 100,
                "page": page,
            }

            try:
                data = self._make_request("questions", params)
            except Exception as e:
                self.logger.error(f"Error fetching questions: {e}")
                break

            for question in data.get("items", []):
                if question.get("score", 0) >= self.min_score:
                    yield question

            if not data.get("has_more", False):
                break

            page += 1

    def _get_best_answer(self, question_id: int) -> Optional[Dict]:
        """Get accepted or highest-voted answer"""
        params = {
            "order": "desc",
            "sort": "votes",
        }

        try:
            data = self._make_request(f"questions/{question_id}/answers", params)
            answers = data.get("items", [])

            # Prefer accepted answer
            for answer in answers:
                if answer.get("is_accepted", False) and answer.get("score", 0) >= 0:
                    return answer

            # Fall back to highest voted
            if answers and answers[0].get("score", 0) >= 0:
                return answers[0]

            return None
        except Exception as e:
            self.logger.warning(f"Error fetching answer for Q{question_id}: {e}")
            return None

    def transform(self, raw_item: Dict) -> Optional[QAPair]:
        """Transform Q&A into training format"""
        question = raw_item["question"]
        answer = raw_item["answer"]
        tag = raw_item["tag"]

        # Clean HTML to markdown
        q_title = question.get("title", "")
        q_body = self._clean_html(question.get("body", ""))
        a_body = self._clean_html(answer.get("body", ""))

        # Skip if answer is too short
        if len(a_body) < 100:
            return None

        # Create instruction from title
        instruction = self._title_to_instruction(q_title)

        # Use question body as input if substantial
        input_text = q_body if len(q_body) > 50 else ""

        return QAPair(
            instruction=instruction,
            input=input_text,
            output=a_body,
            source=f"stackoverflow:{question['question_id']}",
            source_type="stackoverflow",
            score=answer.get("score", 0),
            tags=question.get("tags", []),
            url=question.get("link", ""),
            metadata={
                "question_score": question.get("score", 0),
                "answer_score": answer.get("score", 0),
                "is_accepted": answer.get("is_accepted", False),
                "primary_tag": tag,
            }
        )

    def _clean_html(self, text: str) -> str:
        """Convert HTML to clean markdown"""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Convert code blocks
        text = re.sub(
            r'<pre><code[^>]*>(.*?)</code></pre>',
            r'```\n\1\n```',
            text,
            flags=re.DOTALL
        )
        text = re.sub(r'<code>(.*?)</code>', r'`\1`', text)

        # Convert lists
        text = re.sub(r'<li>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
        text = re.sub(r'<[ou]l>', '', text)
        text = re.sub(r'</[ou]l>', '', text)

        # Convert paragraphs and breaks
        text = re.sub(r'<p>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
        text = re.sub(r'<br\s*/?>', '\n', text)

        # Convert links
        text = re.sub(r'<a href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text)

        # Convert headers
        text = re.sub(r'<h1>(.*?)</h1>', r'# \1\n', text)
        text = re.sub(r'<h2>(.*?)</h2>', r'## \1\n', text)
        text = re.sub(r'<h3>(.*?)</h3>', r'### \1\n', text)

        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def _title_to_instruction(self, title: str) -> str:
        """Convert question title to natural instruction"""
        title = title.strip()

        # If already a question, use as-is
        if title.endswith("?"):
            return title

        # Common question starters
        starters = [
            "How do I ", "How to ", "What is ", "Why does ",
            "Can I ", "Should I ", "Is it possible to "
        ]

        for starter in starters:
            if title.lower().startswith(starter.lower()):
                return title + "?"

        # Default: make it a question
        return f"How do I {title.lower()}?"


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stack Exchange Collector")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--tags", nargs="+", help="Tags to collect")
    parser.add_argument("--max-per-tag", type=int, default=200, help="Max items per tag")
    parser.add_argument("--min-score", type=int, default=10, help="Minimum question score")
    parser.add_argument("--api-key", help="Stack Exchange API key")
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "full"], default="alpaca")

    args = parser.parse_args()

    config = {
        "min_score": args.min_score,
        "max_per_tag": args.max_per_tag,
    }
    if args.tags:
        config["tags"] = args.tags
    if args.api_key:
        config["api_key"] = args.api_key

    collector = StackExchangeCollector(config)
    collector.run(args.output, format=args.format)
