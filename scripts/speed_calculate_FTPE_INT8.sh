#!/usr/bin/env bash
# Benchmark the FT_PE INT8+CRL engine (T=3 canonical) on the current GPU.
#
# Assumes the deploy has already produced an INT8 engine at the path under
# assets/QAT/. To produce one, run:
#
#   bash src/FTPE_INT8/scripts/run_on_a4000.sh
#
# (See src/FTPE_INT8/README.md for the full deploy checklist.)
#
# Override defaults via env vars (BATCH, FRAMES, WARMUP, ITERS, FTPE_INT8_ENGINE,
# FTPE_BF16_ENGINE) or pass extra args after `--`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FTPE_INT8_ENGINE=${FTPE_INT8_ENGINE:-assets/QAT/int8_dyn_crl_t3.engine}
FTPE_BF16_ENGINE=${FTPE_BF16_ENGINE:-assets/model/FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine}
FTPE_TEXT_FEATURES=${FTPE_TEXT_FEATURES:-assets/model/FT_text_features.json}

BATCH=${BATCH:-16}
FRAMES=${FRAMES:-3}      # T=3 is the canonical FT-T3 INT8 deploy
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}

PYTHON=${PYTHON:-python3}

echo "[FT_PE_INT8] INT8 engine:        $FTPE_INT8_ENGINE"
echo "[FT_PE_INT8] BF16 bootstrap:     $FTPE_BF16_ENGINE"
echo "[FT_PE_INT8] text features:      $FTPE_TEXT_FEATURES"
echo "[FT_PE_INT8] batch=$BATCH frames=$FRAMES warmup=$WARMUP iters=$ITERS"

if [ ! -f "$FTPE_INT8_ENGINE" ]; then
    echo "[FT_PE_INT8] ERR: INT8 engine not found: $FTPE_INT8_ENGINE" >&2
    echo "[FT_PE_INT8]      Build it via src/FTPE_INT8/scripts/run_on_a4000.sh" >&2
    echo "[FT_PE_INT8]      (see src/FTPE_INT8/README.md)" >&2
    exit 1
fi

EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

"$PYTHON" src/FTPE_INT8/speed_calculate_FTPE_INT8.py \
    --engine "$FTPE_INT8_ENGINE" \
    --ftpe-engine "$FTPE_BF16_ENGINE" \
    --text-features "$FTPE_TEXT_FEATURES" \
    --batch "$BATCH" \
    --frames "$FRAMES" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    "${EXTRA[@]}"
