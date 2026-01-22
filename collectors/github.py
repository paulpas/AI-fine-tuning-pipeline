#!/usr/bin/env python3
"""
GitHub Collector

Collects training data from GitHub repositories:
- Issues with helpful comments/solutions
- Discussions (Q&A format)
- Code examples from READMEs and examples/

Features:
- Search for repositories by query (e.g., "python kubernetes")
- Fetches closed issues with good solutions
- Extracts Q&A from GitHub Discussions
- Collects code examples from documentation
- Respects rate limits and handles pagination

Configuration:
    token: GitHub personal access token
    repos: List of repos to collect from (e.g., ["kubernetes-client/python"])
    searches: List of search queries to find repos (e.g., ["python devops", "terraform aws"])
    max_repos_per_search: Maximum repos per search query (default: 100)
    min_stars: Minimum stars for searched repos (default: 100)
    max_issues_per_repo: Maximum issues per repo (default: 50)
    collect_discussions: Whether to collect discussions (default: True)
    collect_code: Whether to collect code examples (default: True)

Usage:
    python -m collectors.github --output data/training/
    python -m collectors.github --search "python kubernetes" --search "terraform aws"

Environment Variables:
    GITHUB_TOKEN: Personal access token for API access
"""

import os
import re
import time
import base64
from typing import Dict, List, Optional, Generator

import requests

from .base import BaseCollector, QAPair
from .registry import register_collector


@register_collector
class GitHubCollector(BaseCollector):
    """
    Collector for GitHub issues, discussions, and code examples.
    """

    BASE_URL = "https://api.github.com"

    # Default DevOps-focused repositories
    DEFAULT_REPOS = [
        "kubernetes-client/python",
        "docker/docker-py",
        "hashicorp/terraform",
        "ansible/ansible",
        "pytest-dev/pytest",
        "tiangolo/fastapi",
        "pallets/flask",
        "boto/boto3",
        "prometheus/client_python",
    ]

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.session = requests.Session()
        self._request_count = 0
        self._last_request_time = 0

    def get_name(self) -> str:
        return "github"

    def get_description(self) -> str:
        return "Collects issues, discussions, and code from GitHub repositories"

    def get_required_config_keys(self) -> List[str]:
        return []  # Token is optional but recommended

    def get_optional_config_keys(self) -> Dict:
        return {
            "token": None,
            "repos": self.DEFAULT_REPOS,
            "searches": [],  # List of search queries to find repos
            "max_repos_per_search": 100,  # Max repos per search query
            "min_stars": 100,  # Minimum stars for searched repos
            "max_issues_per_repo": 50,
            "max_discussions_per_repo": 50,
            "collect_issues": True,
            "collect_discussions": True,
            "collect_code": False,  # Can be slow
            "request_delay": 0.5,
        }

    def setup(self) -> None:
        """Initialize API token from config or environment"""
        self.token = self.config.get("token") or os.environ.get("GITHUB_TOKEN")
        self.repos = list(self.config.get("repos", self.DEFAULT_REPOS))  # Copy to avoid mutation
        self.searches = self.config.get("searches", [])
        self.max_repos_per_search = self.config.get("max_repos_per_search", 100)
        self.min_stars = self.config.get("min_stars", 100)
        self.max_issues = self.config.get("max_issues_per_repo", 50)
        self.max_discussions = self.config.get("max_discussions_per_repo", 50)
        self.collect_issues = self.config.get("collect_issues", True)
        self.collect_discussions = self.config.get("collect_discussions", True)
        self.collect_code = self.config.get("collect_code", False)
        self.request_delay = self.config.get("request_delay", 0.5)

        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            self.logger.info("Using GitHub token")
        else:
            self.logger.warning("No token - rate limits will be strict (60/hour)")

        self.session.headers["Accept"] = "application/vnd.github.v3+json"

        # Perform searches and add repos
        if self.searches:
            searched_repos = self._search_repositories()
            self.logger.info(f"Found {len(searched_repos)} repos from {len(self.searches)} search queries")
            # Add searched repos (avoiding duplicates)
            existing = set(self.repos)
            for repo in searched_repos:
                if repo not in existing:
                    self.repos.append(repo)
                    existing.add(repo)

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        if not url.startswith("http"):
            url = f"{self.BASE_URL}/{url}"

        response = self.session.get(url, params=params)
        self._last_request_time = time.time()
        self._request_count += 1

        # Handle rate limiting
        if response.status_code == 403:
            remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            if remaining == 0:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset_time - time.time(), 60)
                self.logger.warning(f"Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
                return self._make_request(url, params)

        response.raise_for_status()
        return response.json()

    def _search_repositories(self) -> List[str]:
        """Search GitHub for repositories matching configured queries"""
        all_repos = []
        seen = set()

        for query in self.searches:
            self.logger.info(f"Searching repositories: {query}")

            # Build search query with stars filter
            search_query = f"{query} stars:>={self.min_stars}"

            page = 1
            repos_from_query = 0

            while repos_from_query < self.max_repos_per_search:
                params = {
                    "q": search_query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(100, self.max_repos_per_search - repos_from_query),
                    "page": page,
                }

                try:
                    data = self._make_request("search/repositories", params)
                except Exception as e:
                    self.logger.error(f"Error searching repos: {e}")
                    break

                items = data.get("items", [])
                if not items:
                    break

                for repo in items:
                    full_name = repo.get("full_name")
                    if full_name and full_name not in seen:
                        seen.add(full_name)
                        all_repos.append(full_name)
                        repos_from_query += 1

                        if repos_from_query >= self.max_repos_per_search:
                            break

                # Check if more pages available
                total_count = data.get("total_count", 0)
                if page * 100 >= total_count or page * 100 >= 1000:  # GitHub limits to 1000 results
                    break

                page += 1

            self.logger.info(f"  Found {repos_from_query} repos for query: {query}")

        return all_repos

    def collect(self) -> Generator[Dict, None, None]:
        """Collect data from configured repositories"""
        for repo in self.repos:
            self.logger.info(f"Collecting from: {repo}")

            # Collect issues
            if self.collect_issues:
                for item in self._collect_issues(repo):
                    yield item

            # Collect discussions
            if self.collect_discussions:
                for item in self._collect_discussions(repo):
                    yield item

            # Collect code examples
            if self.collect_code:
                for item in self._collect_code_examples(repo):
                    yield item

    def _collect_issues(self, repo: str) -> Generator[Dict, None, None]:
        """Collect closed issues with helpful solutions"""
        params = {
            "state": "closed",
            "sort": "comments",
            "direction": "desc",
            "per_page": min(self.max_issues, 100),
        }

        try:
            issues = self._make_request(f"repos/{repo}/issues", params)
        except Exception as e:
            self.logger.error(f"Error fetching issues from {repo}: {e}")
            return

        count = 0
        for issue in issues:
            if count >= self.max_issues:
                break

            # Skip pull requests
            if issue.get("pull_request"):
                continue

            # Get comments
            comments_url = issue.get("comments_url")
            if not comments_url or issue.get("comments", 0) == 0:
                continue

            try:
                comments = self._make_request(comments_url)
            except Exception:
                continue

            # Find the best comment (most reactions)
            best_comment = self._find_best_comment(comments)
            if best_comment and len(best_comment.get("body", "")) > 100:
                yield {
                    "type": "issue",
                    "repo": repo,
                    "issue": issue,
                    "solution": best_comment,
                }
                count += 1

        self.logger.info(f"  Issues: {count} collected")

    def _find_best_comment(self, comments: List[Dict]) -> Optional[Dict]:
        """Find the most helpful comment"""
        best = None
        best_score = -1

        for comment in comments:
            reactions = comment.get("reactions", {})
            score = (
                reactions.get("+1", 0) * 2 +
                reactions.get("heart", 0) * 2 +
                reactions.get("hooray", 0) +
                reactions.get("rocket", 0) -
                reactions.get("-1", 0) * 2
            )

            # Also consider comment length (prefer substantial answers)
            body_len = len(comment.get("body", ""))
            if body_len > 200:
                score += 1

            if score > best_score:
                best_score = score
                best = comment

        return best

    def _collect_discussions(self, repo: str) -> Generator[Dict, None, None]:
        """Collect Q&A from GitHub Discussions (GraphQL API)"""
        # Discussions require GraphQL API
        if not self.token:
            return

        query = """
        query($owner: String!, $name: String!, $first: Int!) {
            repository(owner: $owner, name: $name) {
                discussions(first: $first, orderBy: {field: CREATED_AT, direction: DESC}) {
                    nodes {
                        title
                        body
                        answer {
                            body
                            author { login }
                            upvoteCount
                        }
                        category { name }
                        url
                    }
                }
            }
        }
        """

        owner, name = repo.split("/")
        variables = {
            "owner": owner,
            "name": name,
            "first": min(self.max_discussions, 100),
        }

        try:
            response = self.session.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.logger.debug(f"Discussions not available for {repo}: {e}")
            return

        discussions = data.get("data", {}).get("repository", {}).get("discussions", {}).get("nodes", [])

        count = 0
        for disc in discussions:
            if disc.get("answer") and disc["answer"].get("body"):
                yield {
                    "type": "discussion",
                    "repo": repo,
                    "discussion": disc,
                }
                count += 1

        if count > 0:
            self.logger.info(f"  Discussions: {count} collected")

    def _collect_code_examples(self, repo: str) -> Generator[Dict, None, None]:
        """Collect code examples from examples/ directory"""
        # Try common example directories
        example_paths = ["examples", "docs/examples", "samples", "demo"]

        for path in example_paths:
            try:
                contents = self._make_request(f"repos/{repo}/contents/{path}")
                if isinstance(contents, list):
                    for item in contents:
                        if item.get("name", "").endswith(".py"):
                            yield {
                                "type": "code",
                                "repo": repo,
                                "file": item,
                                "path": f"{path}/{item['name']}",
                            }
            except Exception:
                continue

    def transform(self, raw_item: Dict) -> Optional[QAPair]:
        """Transform raw data into training format"""
        item_type = raw_item.get("type")

        if item_type == "issue":
            return self._transform_issue(raw_item)
        elif item_type == "discussion":
            return self._transform_discussion(raw_item)
        elif item_type == "code":
            return self._transform_code(raw_item)

        return None

    def _transform_issue(self, raw_item: Dict) -> Optional[QAPair]:
        """Transform issue into Q&A format"""
        issue = raw_item["issue"]
        solution = raw_item["solution"]
        repo = raw_item["repo"]

        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        answer = solution.get("body", "")

        # Skip if answer is too short
        if len(answer) < 100:
            return None

        # Truncate long issue bodies
        if len(body) > 500:
            body = body[:500] + "..."

        return QAPair(
            instruction=title,
            input=body,
            output=answer,
            source=f"github:{repo}:{issue['number']}",
            source_type="github",
            score=solution.get("reactions", {}).get("+1", 0),
            tags=[repo.split("/")[0]],
            url=issue.get("html_url", ""),
            metadata={
                "repo": repo,
                "issue_number": issue["number"],
                "comment_reactions": solution.get("reactions", {}),
            }
        )

    def _transform_discussion(self, raw_item: Dict) -> Optional[QAPair]:
        """Transform discussion into Q&A format"""
        disc = raw_item["discussion"]
        repo = raw_item["repo"]

        title = disc.get("title", "")
        body = disc.get("body", "") or ""
        answer = disc.get("answer", {}).get("body", "")

        if len(answer) < 100:
            return None

        if len(body) > 500:
            body = body[:500] + "..."

        return QAPair(
            instruction=title,
            input=body,
            output=answer,
            source=f"github:{repo}:discussion",
            source_type="github",
            score=disc.get("answer", {}).get("upvoteCount", 0),
            tags=[repo.split("/")[0], "discussion"],
            url=disc.get("url", ""),
            metadata={
                "repo": repo,
                "category": disc.get("category", {}).get("name", ""),
            }
        )

    def _transform_code(self, raw_item: Dict) -> Optional[QAPair]:
        """Transform code example into training format"""
        file_info = raw_item["file"]
        repo = raw_item["repo"]
        path = raw_item["path"]

        # Fetch file content
        try:
            content_data = self._make_request(file_info["url"])
            content = base64.b64decode(content_data.get("content", "")).decode("utf-8")
        except Exception:
            return None

        if len(content) < 50 or len(content) > 10000:
            return None

        # Extract docstring or first comment as description
        description = self._extract_description(content)
        if not description:
            description = f"Example from {repo}: {file_info['name']}"

        return QAPair(
            instruction=f"Show me a Python example for {description}",
            input="",
            output=f"```python\n{content}\n```",
            source=f"github:{repo}:{path}",
            source_type="github",
            score=0,
            tags=[repo.split("/")[0], "example"],
            url=file_info.get("html_url", ""),
            metadata={
                "repo": repo,
                "path": path,
            }
        )

    def _extract_description(self, code: str) -> Optional[str]:
        """Extract description from docstring or comments"""
        # Try module docstring
        match = re.match(r'^"""(.*?)"""', code, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n")[0]

        match = re.match(r"^'''(.*?)'''", code, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n")[0]

        # Try first comment
        match = re.match(r'^#\s*(.+)$', code, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return None


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GitHub Collector")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--repos", nargs="+", help="Repos to collect from")
    parser.add_argument("--search", action="append", dest="searches",
                        help="Search query to find repos (can be used multiple times)")
    parser.add_argument("--max-repos", type=int, default=100,
                        help="Max repos per search query (default: 100)")
    parser.add_argument("--min-stars", type=int, default=100,
                        help="Minimum stars for searched repos (default: 100)")
    parser.add_argument("--max-issues", type=int, default=50, help="Max issues per repo")
    parser.add_argument("--token", help="GitHub token")
    parser.add_argument("--no-discussions", action="store_true", help="Skip discussions")
    parser.add_argument("--collect-code", action="store_true", help="Collect code examples")
    parser.add_argument("--no-default-repos", action="store_true",
                        help="Don't include default repos (only use search results)")
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "full"], default="alpaca")

    args = parser.parse_args()

    config = {
        "max_issues_per_repo": args.max_issues,
        "max_repos_per_search": args.max_repos,
        "min_stars": args.min_stars,
        "collect_discussions": not args.no_discussions,
        "collect_code": args.collect_code,
    }

    # Handle repos - either from args, or empty if no-default-repos
    if args.repos:
        config["repos"] = args.repos
    elif args.no_default_repos:
        config["repos"] = []

    # Add search queries
    if args.searches:
        config["searches"] = args.searches

    if args.token:
        config["token"] = args.token

    collector = GitHubCollector(config)
    collector.run(args.output, format=args.format)
