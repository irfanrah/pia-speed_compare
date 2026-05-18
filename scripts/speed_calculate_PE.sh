#!/usr/bin/env bash
# Benchmark PE-Core-L14-336 TRT engine on the current GPU.
#
# - Downloads ONNX from HuggingFace if missing
# - Exports a TRT engine if missing
# - Runs src/speed_calculate_PE.py
#
# Override defaults via env vars (PE_HF_REPO, PE_HF_FILE, PE_ONNX, PE_ENGINE,
# BATCH, WARMUP, ITERS) or pass extra args after `--` (forwarded to the python
# benchmark, e.g. `./scripts/speed_calculate_PE.sh -- --tag a4000_run1`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PE_HF_REPO=${PE_HF_REPO:-PIA-SPACE-LAB/PE-Core-L14-336}
PE_HF_FILE=${PE_HF_FILE:-PE-Core-L14-336.onnx}
PE_ONNX=${PE_ONNX:-assets/model/PE-Core-L14-336.onnx}
PE_ENGINE=${PE_ENGINE:-assets/model/PE-Core-L14-336.engine}
PE_TEXT_FEATURES=${PE_TEXT_FEATURES:-assets/model/text_features.json}

BATCH=${BATCH:-8}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}

PYTHON=${PYTHON:-python3}

echo "[PE] engine: $PE_ENGINE"
echo "[PE] text features: $PE_TEXT_FEATURES"
echo "[PE] batch=$BATCH warmup=$WARMUP iters=$ITERS"

# Always run prepare so text_features.json is fetched even when the engine
# already exists; prepare_engine.py short-circuits the engine build itself.
"$PYTHON" src/prepare_engine.py \
    --kind pe \
    --hf-repo "$PE_HF_REPO" \
    --hf-file "$PE_HF_FILE" \
    --onnx "$PE_ONNX" \
    --engine "$PE_ENGINE" \
    --extra-hf-file "text_features.json:$PE_TEXT_FEATURES"

# Forward any extra args after `--` to the python benchmark.
EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

"$PYTHON" src/speed_calculate_PE.py \
    --engine "$PE_ENGINE" \
    --text-features "$PE_TEXT_FEATURES" \
    --batch "$BATCH" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    "${EXTRA[@]}"
