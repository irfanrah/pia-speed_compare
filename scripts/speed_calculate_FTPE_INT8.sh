#!/usr/bin/env bash
# Benchmark the FT_PE INT8+CRL engine (T=3 canonical) on the current GPU.
#
# Assumes the deploy has already produced an INT8 engine at the path under
# assets/QAT/ftpe/engines/. To produce one, run:
#
#   VARIANT=ftpe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh
#
# That builds a dynamic-batch engine (default profile B=4..128 opt=16, T=3)
# that this bench can drive at any BT inside the profile.
#
# Override defaults via env vars (BATCH, FRAMES, WARMUP, ITERS, FTPE_INT8_ENGINE,
# FTPE_BF16_ENGINE) or pass extra args after `--`.
#
# After the speed bench, this wrapper also runs the random-image cos/MSE
# comparator (src/FTPE_INT8/scripts/test_int8_random.py) against every
# ``.engine`` in the same directory as ``FTPE_INT8_ENGINE``. Skip via
# ``SKIP_COS=1``. The cos test needs the QAT-deployed FP32 .pt; default
# location matches what ``run_int8_pipeline.sh`` auto-fetches from HF:
#   FTPE_QAT_PT=src/FTPE_INT8/.hf_cache/ftpe/splitqkv_qat_t3/qat_deploy_fp32.pt
# Override with ``FTPE_QAT_PT=/path/to/qat_deploy_fp32.pt``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FTPE_INT8_ENGINE=${FTPE_INT8_ENGINE:-assets/QAT/ftpe/engines/int8_ftpe_dyn_b4-128_t3_crl.engine}
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
    echo "[FT_PE_INT8]      Build it via" >&2
    echo "[FT_PE_INT8]        VARIANT=ftpe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh" >&2
    echo "[FT_PE_INT8]      Or set FTPE_INT8_ENGINE=/path/to/your/engine.engine" >&2
    exit 1
fi
if [ ! -f "$FTPE_BF16_ENGINE" ]; then
    echo "[FT_PE_INT8] ERR: BF16 bootstrap engine not found: $FTPE_BF16_ENGINE" >&2
    echo "[FT_PE_INT8]      The bench bootstraps FTPEService with the BF16 engine then" >&2
    echo "[FT_PE_INT8]      swaps in the INT8 adapter. Pre-place the BF16 engine or run" >&2
    echo "[FT_PE_INT8]      scripts/speed_calculate_FTPE.sh once to materialise it." >&2
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

# ── Random-image cos/MSE check vs PT BF16 ───────────────────────────────
SKIP_COS=${SKIP_COS:-0}
if [ "$SKIP_COS" = "1" ]; then
    echo "[FT_PE_INT8] cos/MSE check skipped (SKIP_COS=1)"
    exit 0
fi

FTPE_QAT_PT=${FTPE_QAT_PT:-src/FTPE_INT8/.hf_cache/ftpe/splitqkv_qat_t3/qat_deploy_fp32.pt}
COS_ITERS=${COS_ITERS:-5}
COS_MIN_COS=${COS_MIN_COS:-0.99}
COS_MAX_MSE=${COS_MAX_MSE:-1e-3}

if [ ! -f "$FTPE_QAT_PT" ]; then
    echo "[FT_PE_INT8] cos/MSE check skipped — FTPE_QAT_PT not found: $FTPE_QAT_PT" >&2
    echo "[FT_PE_INT8]   Run VARIANT=ftpe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh to fetch it," >&2
    echo "[FT_PE_INT8]   or set FTPE_QAT_PT=/path/to/qat_deploy_fp32.pt, or SKIP_COS=1 to opt out." >&2
    exit 0
fi

FTPE_INT8_ENGINE_DIR="$(dirname "$FTPE_INT8_ENGINE")"

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
# dynamic-profile context (FT_PE's dynamic engine pre-allocates ~6 GB
# at the OPT shape; teardown can lag a few hundred ms after exit).
sleep 2

echo
echo "[FT_PE_INT8] === random-image cos/MSE vs PT BF16 ==="
echo "[FT_PE_INT8] engines:    $FTPE_INT8_ENGINE_DIR"
echo "[FT_PE_INT8] PT ckpt:    $FTPE_QAT_PT"
echo "[FT_PE_INT8] B=$BATCH  T=$FRAMES  iters=$COS_ITERS  min_cos=$COS_MIN_COS  max_mse=$COS_MAX_MSE"

# `test_int8_random.py` walks every .engine in --engine-dir; the per-engine
# profile check skips engines whose [min,max] doesn't include BT.
COS_OUT_DIR=${COS_OUT_DIR:-results}
COS_TAG=${COS_TAG:-ftpe_int8}
"$PYTHON" src/FTPE_INT8/scripts/test_int8_random.py \
    --engine-dir "$FTPE_INT8_ENGINE_DIR" \
    --ft_ckpt "$FTPE_QAT_PT" \
    --config_name PE-Core-L14-336 \
    --batch_videos "$BATCH" --frames_per_video "$FRAMES" \
    --iters "$COS_ITERS" \
    --min_cos "$COS_MIN_COS" \
    --max_mse "$COS_MAX_MSE" \
    --out-dir "$COS_OUT_DIR" --tag "$COS_TAG" || {
        echo "[FT_PE_INT8] cos/MSE check exited non-zero (engine(s) below threshold)" >&2
        exit 0  # don't fail the wrapper — the speed numbers above are still valid
    }
