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
FTPE_TEXT_FEATURES=${FTPE_TEXT_FEATURES:-assets/model/FT_text_features.json}

BATCH=${BATCH:-8}
FRAMES=${FRAMES:-1}    # T: 1, 3, or 8 are common (any value within engine profile)
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}

PYTHON=${PYTHON:-python3}

echo "[FT_PE] engine: $FTPE_ENGINE"
echo "[FT_PE] text features: $FTPE_TEXT_FEATURES"
echo "[FT_PE] batch=$BATCH frames=$FRAMES warmup=$WARMUP iters=$ITERS"

# Build a wide engine so the same .engine supports B=1..16 and T=1..8 without
# rebuilding. Workspace stays bounded; max=(16,8) on a 16 GB card uses ~10 GB
# during build. If you change MAX_BATCH/MAX_FRAMES you must delete the engine
# file to trigger a rebuild.
MAX_BATCH=${MAX_BATCH:-16}
MAX_FRAMES=${MAX_FRAMES:-8}
OPT_FRAMES=${OPT_FRAMES:-$FRAMES}
# Sanity: if caller asks for BATCH > MAX_BATCH, widen MAX_BATCH.
if [ "$BATCH" -gt "$MAX_BATCH" ]; then
    MAX_BATCH=$BATCH
fi
if [ "$FRAMES" -gt "$MAX_FRAMES" ]; then
    MAX_FRAMES=$FRAMES
fi
# Refuse silent profile mismatch: detect existing engine that's too narrow.
if [ -f "$FTPE_ENGINE" ]; then
    # Engines are tagged by their build profile only at build time -- we can't
    # introspect cheaply from shell, so trust the user. Document the override.
    :
fi
# Always run prepare so FT_text_features.json is fetched even when the engine
# already exists.
"$PYTHON" src/prepare_engine.py \
    --kind ftpe \
    --hf-repo "$FTPE_HF_REPO" \
    --hf-file "$FTPE_HF_FILE" \
    --onnx "$FTPE_ONNX" \
    --engine "$FTPE_ENGINE" \
    --opt-batch "$BATCH" --max-batch "$MAX_BATCH" \
    --opt-frames "$OPT_FRAMES" --max-frames "$MAX_FRAMES" \
    --extra-hf-file "FT_text_features.json:$FTPE_TEXT_FEATURES"

EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

"$PYTHON" src/speed_calculate_FTPE.py \
    --engine "$FTPE_ENGINE" \
    --text-features "$FTPE_TEXT_FEATURES" \
    --batch "$BATCH" \
    --frames "$FRAMES" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    "${EXTRA[@]}"
