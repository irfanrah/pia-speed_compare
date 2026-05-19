#!/usr/bin/env bash
# Benchmark the zero-shot PE INT8(+CRL) engine on the current GPU.
#
# Drives ``src/speed_calculate_PE.py`` (the same five-stage harness used by
# the BF16 PE bench) against an INT8 TRT engine that the PE production
# service can load. The engine must accept a B=1 input at runtime — that's
# what ``PEService._init_default_values`` probes at startup to compute the
# zero-mask vector. A *static* B=16 engine fails the init even though it
# would otherwise run the bench shape just fine. Use a dynamic-profile
# engine whose [min, max] includes both 1 and your bench BATCH.
#
# Pipeline produces a static B=16 engine by default (deploy candidate). To
# build a dynamic test engine from the same clean INT8 ONNX:
#
#   python3 src/FTPE_INT8/pe_int8/build_dynamic_engine.py \
#       --onnx assets/QAT/pe/onnx/int8_pe_b16t1_clean.onnx \
#       --save_engine assets/QAT/pe/engines/int8_pe_dyn_b1-16_t1.engine \
#       --min_shape input:1x3x336x336 \
#       --opt_shape input:16x3x336x336 \
#       --max_shape input:16x3x336x336 \
#       --mode strongly_typed
#
# Override defaults via env vars (BATCH, WARMUP, ITERS, PE_INT8_ENGINE,
# PE_TEXT_FEATURES) or pass extra args after ``--``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PE_INT8_ENGINE=${PE_INT8_ENGINE:-assets/QAT/pe/engines/int8_pe_dyn_b1-16_t1.engine}
PE_TEXT_FEATURES=${PE_TEXT_FEATURES:-assets/model/text_features.json}

BATCH=${BATCH:-16}       # zero-shot PE INT8 deploy ships at B=16
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}

PYTHON=${PYTHON:-python3}

echo "[PE_INT8] INT8 engine:    $PE_INT8_ENGINE"
echo "[PE_INT8] text features:  $PE_TEXT_FEATURES"
echo "[PE_INT8] batch=$BATCH warmup=$WARMUP iters=$ITERS"

if [ ! -f "$PE_INT8_ENGINE" ]; then
    echo "[PE_INT8] ERR: INT8 engine not found: $PE_INT8_ENGINE" >&2
    echo "[PE_INT8]      Build the static deploy engine via" >&2
    echo "[PE_INT8]        VARIANT=pe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh" >&2
    echo "[PE_INT8]      then build a dynamic test engine from the resulting clean ONNX" >&2
    echo "[PE_INT8]      (see the comment block at the top of this script for the exact" >&2
    echo "[PE_INT8]      build_dynamic_engine.py command). Or set" >&2
    echo "[PE_INT8]      PE_INT8_ENGINE=/path/to/your/engine.engine" >&2
    exit 1
fi
if [ ! -f "$PE_TEXT_FEATURES" ]; then
    echo "[PE_INT8] ERR: text features not found: $PE_TEXT_FEATURES" >&2
    echo "[PE_INT8]      Run scripts/speed_calculate_PE.sh once to materialise it," >&2
    echo "[PE_INT8]      or set PE_TEXT_FEATURES=/path/to/text_features.json" >&2
    exit 1
fi

EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

"$PYTHON" src/speed_calculate_PE.py \
    --engine "$PE_INT8_ENGINE" \
    --text-features "$PE_TEXT_FEATURES" \
    --batch "$BATCH" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --tag int8 \
    "${EXTRA[@]}"
