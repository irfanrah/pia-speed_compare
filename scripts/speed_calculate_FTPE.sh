#!/usr/bin/env bash
# Benchmark FT_PE (fine-tuned PE) TRT engine on the current GPU.
#
# - Downloads ONNX from HuggingFace if missing
# - Exports a TRT engine with temporal dim if missing
# - Runs src/speed_calculate_FTPE.py
#
# The HF repo/file are NOT publicly known; set FTPE_HF_REPO and FTPE_HF_FILE
# in the environment (or pre-place the engine at $FTPE_ENGINE).
#
# Override defaults via env vars (FTPE_HF_REPO, FTPE_HF_FILE, FTPE_ONNX,
# FTPE_ENGINE, BATCH, FRAMES, WARMUP, ITERS) or pass extra args after `--`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FTPE_ID=${FTPE_ID:-FT_PE-Core-L14-336_260318}
FTPE_HF_REPO=${FTPE_HF_REPO:-PIA-SPACE-LAB/${FTPE_ID}}
FTPE_HF_FILE=${FTPE_HF_FILE:-${FTPE_ID}_vision_no_mean_pooling.onnx}
FTPE_ONNX=${FTPE_ONNX:-assets/model/${FTPE_ID}_vision_no_mean_pooling.onnx}
FTPE_ENGINE=${FTPE_ENGINE:-assets/model/${FTPE_ID}_vision_no_mean_pooling.engine}

BATCH=${BATCH:-8}
FRAMES=${FRAMES:-1}    # WINDOW_SIZE: 1=FPS_8 mode, 3=FPS_3 mode (see ft_pe/config.py)
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}

PYTHON=${PYTHON:-python3}

echo "[FT_PE] engine: $FTPE_ENGINE"
echo "[FT_PE] batch=$BATCH frames=$FRAMES warmup=$WARMUP iters=$ITERS"

if [ ! -f "$FTPE_ENGINE" ]; then
    echo "[FT_PE] engine missing; downloading + converting ..."
    # Size the engine for the benchmark shape only. The temporal FT_PE model
    # eats GB of workspace if max_batch/max_frames are left at the
    # trt_export.py defaults (32/16) — fine on big GPUs, OOM on a 16 GB card.
    MAX_BATCH=${MAX_BATCH:-$BATCH}
    MAX_FRAMES=${MAX_FRAMES:-$FRAMES}
    "$PYTHON" src/prepare_engine.py \
        --kind ftpe \
        --hf-repo "$FTPE_HF_REPO" \
        --hf-file "$FTPE_HF_FILE" \
        --onnx "$FTPE_ONNX" \
        --engine "$FTPE_ENGINE" \
        --opt-batch "$BATCH" --max-batch "$MAX_BATCH" \
        --opt-frames "$FRAMES" --max-frames "$MAX_FRAMES"
fi

EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

"$PYTHON" src/speed_calculate_FTPE.py \
    --engine "$FTPE_ENGINE" \
    --batch "$BATCH" \
    --frames "$FRAMES" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    "${EXTRA[@]}"
