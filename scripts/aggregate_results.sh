#!/usr/bin/env bash
# Aggregate every results/*.json into a fixed-width text table.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PYTHON=${PYTHON:-python3}
"$PYTHON" src/aggregate_results.py "$@"
