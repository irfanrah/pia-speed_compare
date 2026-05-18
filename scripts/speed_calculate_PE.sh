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
PE_HF_FILE=${PE_HF_FILE:-onnx/PE-Core-L14-336_vision_dynamic.onnx}
PE_ONNX=${PE_ONNX:-assets/model/PE-Core-L14-336_vision_dynamic.onnx}
PE_ENGINE=${PE_ENGINE:-assets/model/PE-Core-L14-336_vision_dynamic.engine}

BATCH=${BATCH:-8}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}

PYTHON=${PYTHON:-python3}

echo "[PE] engine: $PE_ENGINE"
echo "[PE] batch=$BATCH warmup=$WARMUP iters=$ITERS"

if [ ! -f "$PE_ENGINE" ]; then
    echo "[PE] engine missing; downloading + converting ..."
    "$PYTHON" src/prepare_engine.py \
        --kind pe \
        --hf-repo "$PE_HF_REPO" \
        --hf-file "$PE_HF_FILE" \
        --onnx "$PE_ONNX" \
        --engine "$PE_ENGINE"
fi

# Forward any extra args after `--` to the python benchmark.
EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

"$PYTHON" src/speed_calculate_PE.py \
    --engine "$PE_ENGINE" \
    --batch "$BATCH" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    "${EXTRA[@]}"
