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
MAX_DEPTH = 4               # safety guard against runaway recursion
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

# --------------------------------------------------------------
# Crawler state (for resume functionality)
# --------------------------------------------------------------
CRAWLER_STATE_FILE = DATA_DIR / "crawler_state.json"
STATE_SAVE_INTERVAL = 10  # Save state every N pages
