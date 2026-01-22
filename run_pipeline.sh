#!/bin/bash
# LLM Training Data Collection Pipeline
# Reads settings from config/pipeline_config.yaml
# Logs all steps to a single file

set -e
cd "$(dirname "$0")"

CONFIG_FILE="${1:-config/pipeline_config.yaml}"
LOGFILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=========================================="
log "LLM Training Data Collection Pipeline"
log "=========================================="
log "Working directory: $(pwd)"
log "Config file: $CONFIG_FILE"
log "Log file: $LOGFILE"

# Load environment
if [ -f .env ]; then
    source .env
    log "Loaded .env file"
else
    log "WARNING: No .env file found"
fi

# Extract config values using Python
get_config() {
    uv run python -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
$1
"
}

log ""
log "=== STEP 1: Stack Overflow Collector ==="
log "Collecting Q&A from Stack Exchange API..."

# Build stackoverflow config from YAML
SO_CONFIG=$(get_config "
import json
so = config.get('external_sources', {}).get('stackoverflow', {})
if so.get('enabled', False):
    cfg = {
        'tags': so.get('tags', []),
        'min_score': so.get('min_score', 10),
        'max_per_tag': so.get('max_per_tag', 200),
    }
    print(json.dumps(cfg))
else:
    print('DISABLED')
")

if [ "$SO_CONFIG" != "DISABLED" ]; then
    uv run python -m collectors stackoverflow -o data/training/ --format full --config "$SO_CONFIG" 2>&1 | tee -a "$LOGFILE"
else
    log "Stack Overflow collector disabled in config"
fi

log ""
log "=== STEP 2: GitHub Collector ==="
log "Collecting issues and discussions from GitHub..."

# Build github config from YAML
GH_CONFIG=$(get_config "
import json
gh = config.get('external_sources', {}).get('github', {})
if gh.get('enabled', False):
    cfg = {
        'repos': gh.get('repos', []),
        'searches': gh.get('searches', []),
        'max_repos_per_search': gh.get('max_repos_per_search', 100),
        'min_stars': gh.get('min_stars', 100),
        'max_issues_per_repo': gh.get('max_issues_per_repo', 50),
        'collect_discussions': gh.get('collect_discussions', True),
        'collect_code': gh.get('collect_code', False),
    }
    print(json.dumps(cfg))
else:
    print('DISABLED')
")

if [ "$GH_CONFIG" != "DISABLED" ]; then
    log "GitHub config: repos + $(echo "$GH_CONFIG" | uv run python -c "import json,sys; c=json.load(sys.stdin); print(len(c.get('searches',[])), 'search queries')")"
    uv run python -m collectors github -o data/training/ --format full --config "$GH_CONFIG" 2>&1 | tee -a "$LOGFILE"
else
    log "GitHub collector disabled in config"
fi

log ""
log "=== STEP 3: Web Crawler (Optional) ==="
# Check if any web sources are enabled
WEB_ENABLED=$(get_config "
web = config.get('web_sources', [])
enabled = [w for w in web if w.get('enabled', False)]
print('YES' if enabled else 'NO')
")

if [ "$WEB_ENABLED" = "YES" ]; then
    log "Web crawler enabled - requires playwright install"
    # uv run python -m collectors web_crawler -o data/training/ --format full 2>&1 | tee -a "$LOGFILE"
else
    log "Web crawler disabled in config"
fi

log ""
log "=== STEP 4: Summary ==="
log "Output files:"
ls -lh data/training/*.json 2>/dev/null | tee -a "$LOGFILE"

log ""
log "=========================================="
log "Data Collection Pipeline complete!"
log "=========================================="

# Optional: Run model evaluation if models exist
log ""
log "=== STEP 5: Model Evaluation (Optional) ==="

# Get model name from config
MODEL_NAME=$(get_config "
pipeline = config.get('pipeline', {})
export_cfg = config.get('export', {}).get('ollama', {})
name = export_cfg.get('model_name', 'python-expert')
version = pipeline.get('version', 'v1')
print(f'{name}-{version}')
")

# Check if model exists in Ollama
if ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
    log "Found fine-tuned model: $MODEL_NAME"
    log "Running evaluation..."
    uv run python scripts/evaluate_model.py --config "$CONFIG_FILE" 2>&1 | tee -a "$LOGFILE"

    REPORT_FILE="docs/${MODEL_NAME}_training_results.md"
    if [ -f "$REPORT_FILE" ]; then
        log "Evaluation report generated: $REPORT_FILE"
    fi
else
    log "Fine-tuned model '$MODEL_NAME' not found in Ollama"
    log "Skipping evaluation. Run manually after training:"
    log "  uv run python scripts/evaluate_model.py --model <model-name> --base deepseek-r1:1.5b"
fi

log ""
log "=========================================="
log "Pipeline complete!"
log "=========================================="
