#!/bin/bash
# =============================================================================
# Collect Training Data from All Configured Sources
# =============================================================================
# This script reads config/data_sources.yaml and:
# 1. Clones all configured repositories
# 2. Extracts training data from each
# 3. Combines and deduplicates the data
#
# Usage: ./scripts/collect_all_data.sh [--skip-clone] [--skip-extract]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/config/data_sources.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_CLONE=false
SKIP_EXTRACT=false

for arg in "$@"; do
    case $arg in
        --skip-clone)
            SKIP_CLONE=true
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Data Collection Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create directories
mkdir -p repos data/training

# -----------------------------------------------------------------------------
# Define repositories (matches config/data_sources.yaml)
# -----------------------------------------------------------------------------
declare -A REPOS
REPOS["k8s-python"]="https://github.com/kubernetes-client/python"
REPOS["pytest-repo"]="https://github.com/pytest-dev/pytest"
REPOS["advanced_python"]="https://github.com/krother/advanced_python"
REPOS["python_mastery"]="https://github.com/dabeaz-course/python-mastery"
REPOS["amazing_python"]="https://github.com/avinashkranjan/Amazing-Python-Scripts"

# Subdirectories to process (empty means entire repo)
declare -A SUBDIRS
SUBDIRS["k8s-python"]="examples"
SUBDIRS["pytest-repo"]="doc/en"
SUBDIRS["advanced_python"]=""
SUBDIRS["python_mastery"]=""
SUBDIRS["amazing_python"]=""

# -----------------------------------------------------------------------------
# Step 1: Clone Repositories
# -----------------------------------------------------------------------------
if [ "$SKIP_CLONE" = false ]; then
    echo -e "${YELLOW}Step 1: Cloning repositories...${NC}"
    echo ""

    for name in "${!REPOS[@]}"; do
        url="${REPOS[$name]}"
        target="repos/$name"

        if [ -d "$target" ]; then
            echo -e "  ${GREEN}✓${NC} $name (already exists)"
        else
            echo -e "  Cloning $name..."
            git clone --depth 1 "$url" "$target" 2>/dev/null || {
                echo -e "  ${RED}✗${NC} Failed to clone $name"
                continue
            }
            echo -e "  ${GREEN}✓${NC} $name"
        fi
    done
    echo ""
else
    echo -e "${YELLOW}Skipping clone step...${NC}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 2: Extract Training Data
# -----------------------------------------------------------------------------
if [ "$SKIP_EXTRACT" = false ]; then
    echo -e "${YELLOW}Step 2: Extracting training data...${NC}"
    echo ""

    for name in "${!REPOS[@]}"; do
        subdir="${SUBDIRS[$name]}"
        if [ -n "$subdir" ]; then
            input_dir="repos/$name/$subdir"
        else
            input_dir="repos/$name"
        fi
        output_file="data/training/${name}_alpaca.json"

        if [ ! -d "$input_dir" ]; then
            echo -e "  ${RED}✗${NC} $name - directory not found: $input_dir"
            continue
        fi

        echo -e "  Processing $name..."

        # Use appropriate script based on content type
        if [ "$name" = "pytest-repo" ]; then
            python scripts/prepare_pytest_data.py \
                --input-dir "$input_dir" \
                --output "$output_file" 2>/dev/null || true
        else
            python scripts/prepare_k8s_python_data.py \
                --input-dir "$input_dir" \
                --output "$output_file" \
                --include-subdirs 2>/dev/null || true
        fi

        if [ -f "$output_file" ]; then
            count=$(python3 -c "import json; print(len(json.load(open('$output_file'))))" 2>/dev/null || echo "0")
            echo -e "  ${GREEN}✓${NC} $name: $count examples"
        else
            echo -e "  ${RED}✗${NC} $name: extraction failed"
        fi
    done
    echo ""
else
    echo -e "${YELLOW}Skipping extraction step...${NC}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 3: Combine Datasets
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 3: Combining datasets...${NC}"

python3 << 'EOF'
import json
import os
from pathlib import Path

training_dir = Path("data/training")
combined = []

# Find all *_alpaca.json files
for json_file in sorted(training_dir.glob("*_alpaca.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)
            print(f"  {json_file.name}: {len(data)} examples")
            combined.extend(data)
    except Exception as e:
        print(f"  {json_file.name}: ERROR - {e}")

print(f"\n  Total combined: {len(combined)}")

with open("data/training/all_python_combined.json", "w") as f:
    json.dump(combined, f, indent=2)

print("  Saved to: data/training/all_python_combined.json")
EOF

echo ""

# -----------------------------------------------------------------------------
# Step 4: Deduplicate
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Step 4: Deduplicating...${NC}"

python scripts/deduplicate_dataset.py \
    --input "data/training/all_python_combined.json" \
    --output "data/training/all_python_deduped.json" \
    --min-output-length 50

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Collection Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ -f "data/training/all_python_deduped.json" ]; then
    final_count=$(python3 -c "import json; print(len(json.load(open('data/training/all_python_deduped.json'))))")
    echo -e "Final dataset: ${GREEN}$final_count${NC} training examples"
    echo -e "Output file: data/training/all_python_deduped.json"
else
    echo -e "${RED}Error: Final dataset not created${NC}"
fi

echo ""
echo "Next steps:"
echo "  1. Review: head -100 data/training/all_python_deduped.json | python -m json.tool"
echo "  2. Train:  ./scripts/run_pipeline.sh"
