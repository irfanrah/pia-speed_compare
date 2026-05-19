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
#
# After the speed bench, this wrapper also runs the random-image cos/MSE
# comparator (src/FTPE_INT8/scripts/test_int8_random.py) against every
# ``.engine`` in the same directory as ``PE_INT8_ENGINE``. Skip via
# ``SKIP_COS=1``. The cos test needs the QAT-deployed FP32 .pt; default
# location matches what ``run_int8_pipeline.sh`` auto-fetches from HF:
#   PE_QAT_PT=src/FTPE_INT8/.hf_cache/pe/splitqkv_qat/qat_deploy_fp32.pt
# Override with ``PE_QAT_PT=/path/to/qat_deploy_fp32.pt``.

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

# Capture the speed-bench JSON path so the cos pass can append its section
# to it. Use `script` (PTY-preserving) so we don't lose interactive flush
# behaviour, falling back to plain `tee` for portability.
SPEED_LOG="$(mktemp -t pe_int8_speed.XXXXXX.log)"
"$PYTHON" src/speed_calculate_PE.py \
    --engine "$PE_INT8_ENGINE" \
    --text-features "$PE_TEXT_FEATURES" \
    --batch "$BATCH" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --tag int8 \
    "${EXTRA[@]}" 2>&1 | tee "$SPEED_LOG"
SPEED_JSON="$(awk '/^wrote: /{print $2; exit}' "$SPEED_LOG")"
rm -f "$SPEED_LOG"

# ── Random-image cos/MSE check vs PT BF16 ───────────────────────────────
SKIP_COS=${SKIP_COS:-0}
if [ "$SKIP_COS" = "1" ]; then
    echo "[PE_INT8] cos/MSE check skipped (SKIP_COS=1)"
    exit 0
fi

PE_QAT_PT=${PE_QAT_PT:-src/FTPE_INT8/.hf_cache/pe/splitqkv_qat/qat_deploy_fp32.pt}
COS_ITERS=${COS_ITERS:-5}
COS_MIN_COS=${COS_MIN_COS:-0.99}
COS_MAX_MSE=${COS_MAX_MSE:-1e-3}

if [ ! -f "$PE_QAT_PT" ]; then
    echo "[PE_INT8] cos/MSE check skipped — PE_QAT_PT not found: $PE_QAT_PT" >&2
    echo "[PE_INT8]   Run VARIANT=pe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh to fetch it," >&2
    echo "[PE_INT8]   or set PE_QAT_PT=/path/to/qat_deploy_fp32.pt, or SKIP_COS=1 to opt out." >&2
    exit 0
fi

PE_INT8_ENGINE_DIR="$(dirname "$PE_INT8_ENGINE")"

# NOTE: we deliberately do NOT prepend pip's nvidia-cudnn-cu12 dir to
# LD_LIBRARY_PATH here (unlike run_int8_pipeline.sh, which does it for ORT).
# torch already bundles its own libcudnn under ``torch/lib/``; forcing a
# different libcudnn earlier on LD_LIBRARY_PATH triggers a version mismatch
# inside ``nn.Conv2d``'s cuDNN init (CUDNN_STATUS_NOT_INITIALIZED). The cos
# pass uses only torch + TensorRT — both find their own CUDA libs without
# manual help. The cos script itself also disables cuDNN as a belt-and-
# braces guard against wheel/system cuDNN mismatches that show up in some
# conda envs (test_int8_random.py sets ``cudnn.enabled = False``).

# Brief settle: give CUDA a moment to release the bench process's
# dynamic-profile context (it pre-allocates ~6 GB on FT_PE / ~3 GB on PE
# at the OPT shape; teardown can lag a few hundred ms).
sleep 2

echo
echo "[PE_INT8] === random-image cos/MSE vs PT BF16 ==="
echo "[PE_INT8] engines:    $PE_INT8_ENGINE_DIR"
echo "[PE_INT8] PT ckpt:    $PE_QAT_PT"
echo "[PE_INT8] B=$BATCH  T=1  iters=$COS_ITERS  min_cos=$COS_MIN_COS  max_mse=$COS_MAX_MSE"

# `test_int8_random.py` walks every .engine in --engine-dir; the per-engine
# profile check inside it skips engines whose [min,max] doesn't include BT.
# When SPEED_JSON was captured above, the cos pass appends a ``cos_mse``
# section to it in place — single JSON per run instead of a sidecar file.
APPEND_TO_ARGS=()
if [ -n "${SPEED_JSON:-}" ] && [ -f "$SPEED_JSON" ]; then
    APPEND_TO_ARGS=(--append-to "$SPEED_JSON")
fi
"$PYTHON" src/FTPE_INT8/scripts/test_int8_random.py \
    --engine-dir "$PE_INT8_ENGINE_DIR" \
    --ft_ckpt "$PE_QAT_PT" \
    --config_name PE-Core-L14-336 \
    --batch_videos "$BATCH" --frames_per_video 1 \
    --iters "$COS_ITERS" \
    --min_cos "$COS_MIN_COS" \
    --max_mse "$COS_MAX_MSE" \
    "${APPEND_TO_ARGS[@]}" || {
        echo "[PE_INT8] cos/MSE check exited non-zero (engine(s) below threshold)" >&2
        exit 0  # don't fail the wrapper — the speed numbers above are still valid
    }
