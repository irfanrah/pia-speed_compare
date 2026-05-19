#!/usr/bin/env bash
# Plot each results/*.json as a PNG with twin Y-axis (latency / GPU temp).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PYTHON=${PYTHON:-python3}
"$PYTHON" src/plot_results.py "$@"
