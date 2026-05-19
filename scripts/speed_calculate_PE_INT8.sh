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

"$PYTHON" src/speed_calculate_PE.py \
    --engine "$PE_INT8_ENGINE" \
    --text-features "$PE_TEXT_FEATURES" \
    --batch "$BATCH" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --tag int8 \
    "${EXTRA[@]}"

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

# Expose PE vendor + pip-installed CUDA libs the test_int8_random script needs.
# (TRT inference normally doesn't need cuDNN, but `import tensorrt` can pull
# in cuBLAS/CUDA-runtime; if the script ever hits a missing-lib failure on a
# host where torch's bundled libs aren't on LD_LIBRARY_PATH, the discovery
# below picks them up the same way run_int8_pipeline.sh does.)
_NV_LIBS="$("$PYTHON" -c '
import os, importlib, site
libs=[]
roots = []
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    n = os.path.join(sp, "nvidia")
    if os.path.isdir(n) and n not in roots: roots.append(n)
for name in ["cudnn","cublas","cuda_runtime","cufft","curand","cusolver","cusparse","nccl","nvjitlink","cuda_cupti","cuda_nvrtc"]:
    p = None
    try:
        m = importlib.import_module(f"nvidia.{name}")
        if getattr(m, "__file__", None):
            p = os.path.join(os.path.dirname(m.__file__), "lib")
    except Exception: pass
    if not p or not os.path.isdir(p):
        for r in roots:
            cand = os.path.join(r, name, "lib")
            if os.path.isdir(cand): p = cand; break
    if p and os.path.isdir(p) and p not in libs: libs.append(p)
print(":".join(libs))
' 2>/dev/null || true)"
if [ -n "$_NV_LIBS" ]; then
    export LD_LIBRARY_PATH="$_NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo
echo "[PE_INT8] === random-image cos/MSE vs PT BF16 ==="
echo "[PE_INT8] engines:    $PE_INT8_ENGINE_DIR"
echo "[PE_INT8] PT ckpt:    $PE_QAT_PT"
echo "[PE_INT8] B=$BATCH  T=1  iters=$COS_ITERS  min_cos=$COS_MIN_COS  max_mse=$COS_MAX_MSE"

# `test_int8_random.py` walks every .engine in --engine-dir; the per-engine
# profile check inside it skips engines whose [min,max] doesn't include BT.
"$PYTHON" src/FTPE_INT8/scripts/test_int8_random.py \
    --engine-dir "$PE_INT8_ENGINE_DIR" \
    --ft_ckpt "$PE_QAT_PT" \
    --config_name PE-Core-L14-336 \
    --batch_videos "$BATCH" --frames_per_video 1 \
    --iters "$COS_ITERS" \
    --min_cos "$COS_MIN_COS" \
    --max_mse "$COS_MAX_MSE" || {
        echo "[PE_INT8] cos/MSE check exited non-zero (engine(s) below threshold)" >&2
        exit 0  # don't fail the wrapper — the speed numbers above are still valid
    }
