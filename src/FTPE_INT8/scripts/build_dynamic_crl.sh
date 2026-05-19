#!/usr/bin/env bash
# Build a DYNAMIC-batch INT8+CRL TRT engine that handles a wide BT range.
#
# Reuses the FP32 dynamic ONNX produced by run_on_a4000.sh (which uses
# dynamic_axes so the batch dim is already symbolic), runs the CRL pre-pass,
# modelopt PTQ at the OPT batch, surgery + dynamic engine build.
#
# Defaults target B = 4..128 at T=3 (BT 12..384), opt at B=16 (BT=48).
#
# Required env (inherited from run_on_a4000.sh if you re-export them):
#   PE_VENDOR        /path/to/pia-prompt_optimization
#   PT_CKPT          /path/to/qat_deploy_fp32.pt
#   DATASET_ROOT     /path/to/clips (contains val/, test/)
#
# Optional env:
#   FP32_ONNX        existing FP32 dynamic-batch ONNX (default: from
#                    OUT_DIR_STATIC/onnx/fp32_b4t3.onnx after run_on_a4000.sh)
#   OUT_DIR_STATIC   the static run dir from run_on_a4000.sh (where the FP32
#                    ONNX + calibration.npy live). Default: assets/QAT/a4000_run.
#   OUT_DIR_DYN      where dynamic artifacts go. Default: assets/QAT/a4000_run_dyn.
#   T_FRAMES         3 (canonical FT-T3)
#   B_MIN B_OPT B_MAX     4, 16, 128
#   SIGMA_K          2.5
#   GPU              0
#   SKIP_BF16        1 to skip the BF16 dynamic engine build (saves ~3-5 min)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
PE_INT8="$EXP_ROOT/pe_int8"
[[ -d "$PE_INT8" ]] || { echo "ERR: bundle pe_int8/ not found at $PE_INT8" >&2; exit 1; }

: "${PE_VENDOR:?set PE_VENDOR=/path/to/pia-prompt_optimization}"
: "${PT_CKPT:?set PT_CKPT=/path/to/qat_deploy_fp32.pt}"
: "${DATASET_ROOT:?set DATASET_ROOT=/path/to/clips (must contain val/)}"

[[ -d "$PE_VENDOR/src/PE/perception_models" ]] || { echo "ERR: PE_VENDOR layout wrong: $PE_VENDOR" >&2; exit 1; }
[[ -f "$PT_CKPT" ]] || { echo "ERR: PT_CKPT not found: $PT_CKPT" >&2; exit 1; }
[[ -d "$DATASET_ROOT/val" ]] || { echo "ERR: $DATASET_ROOT must contain val/" >&2; exit 1; }

T_FRAMES="${T_FRAMES:-3}"
B_MIN="${B_MIN:-4}"
B_OPT="${B_OPT:-16}"
B_MAX="${B_MAX:-128}"
# Modelopt PTQ runs the model under ORT, whose BFC arena refuses single
# allocations > 226 MB. For ViT-L-14 at 336x336 the GELU/attn intermediate
# crosses that cap once BT>=24. PTQ at BT=12 (B=4 T=3) is the largest shape
# that fits; the engine profile can still go much wider. Override via
# B_CALIB / CALIB_BT if you have a card with a larger ORT arena.
B_CALIB="${B_CALIB:-4}"
SIGMA_K="${SIGMA_K:-2.5}"
GPU="${GPU:-0}"
SKIP_BF16="${SKIP_BF16:-0}"

REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"
OUT_DIR_STATIC="${OUT_DIR_STATIC:-$REPO_ROOT/assets/QAT/a4000_run}"
OUT_DIR_DYN="${OUT_DIR_DYN:-$REPO_ROOT/assets/QAT/a4000_run_dyn}"
FP32_ONNX="${FP32_ONNX:-$OUT_DIR_STATIC/onnx/fp32_b${B_MIN}t${T_FRAMES}.onnx}"
CALIB_NPY="${CALIB_NPY:-$OUT_DIR_STATIC/calib/calibration.npy}"

[[ -f "$FP32_ONNX" ]] || { echo "ERR: FP32 ONNX not found: $FP32_ONNX. Run run_on_a4000.sh first." >&2; exit 1; }
[[ -f "$CALIB_NPY" ]] || { echo "ERR: calibration.npy not found: $CALIB_NPY. Run run_on_a4000.sh first." >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export PE_VENDOR
export PE_PMODELS="$PE_VENDOR/src/PE/perception_models"
export PYTHONPATH="$PE_INT8:$PE_PMODELS:$PE_VENDOR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$EXP_ROOT/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
PY="${PYTHONBIN:-$(command -v python3)}"

# Make CUDA-bound libs from the pip-installed nvidia-* wheels visible to ORT.
# Without this the onnxruntime CUDAExecutionProvider fails to load libcudnn.so.9
# and PTQ silently falls back to CPU, which then OOMs the BFC arena at BT>=48.
_NV_LIBS="$("$PY" -c '
import os, importlib
libs=[]
for mod in ["nvidia.cudnn","nvidia.cublas","nvidia.cuda_runtime","nvidia.cufft","nvidia.curand","nvidia.cusolver","nvidia.cusparse","nvidia.nccl","nvidia.nvjitlink","nvidia.cuda_cupti","nvidia.cuda_nvrtc"]:
    try:
        m = importlib.import_module(mod)
        p = os.path.dirname(m.__file__)+"/lib"
        if os.path.isdir(p): libs.append(p)
    except Exception: pass
print(":".join(libs))
' 2>/dev/null || true)"
if [[ -n "$_NV_LIBS" ]]; then
    export LD_LIBRARY_PATH="$_NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MIN_BT=$((B_MIN * T_FRAMES))
OPT_BT=$((B_OPT * T_FRAMES))
MAX_BT=$((B_MAX * T_FRAMES))
CALIB_BT="${CALIB_BT:-$((B_CALIB * T_FRAMES))}"
TAG="dyn_b${B_MIN}-${B_MAX}_t${T_FRAMES}"
IMG_SIZE=336

MIN_SHAPE="input:${MIN_BT}x3x${IMG_SIZE}x${IMG_SIZE}"
OPT_SHAPE="input:${OPT_BT}x3x${IMG_SIZE}x${IMG_SIZE}"
MAX_SHAPE="input:${MAX_BT}x3x${IMG_SIZE}x${IMG_SIZE}"
CALIB_SHAPE="input:${CALIB_BT}x3x${IMG_SIZE}x${IMG_SIZE}"

mkdir -p "$OUT_DIR_DYN"/{onnx,engines,calib,logs}

echo "[dyn] FP32 ONNX:    $FP32_ONNX"
echo "[dyn] calib npy:    $CALIB_NPY"
echo "[dyn] profile:      min=$MIN_SHAPE"
echo "[dyn]                opt=$OPT_SHAPE"
echo "[dyn]                max=$MAX_SHAPE"
echo "[dyn] B range:      $B_MIN..$B_MAX  (T=$T_FRAMES → BT $MIN_BT..$MAX_BT, opt=$OPT_BT)"
echo "[dyn] calib shape:  $CALIB_SHAPE  (BT=$CALIB_BT, decoupled from engine opt)"
echo "[dyn] SIGMA_K:      $SIGMA_K"
echo "[dyn] GPU:          $GPU   SKIP_BF16=$SKIP_BF16"
echo "[dyn] out dir:      $OUT_DIR_DYN"
echo

BF16_ENG="$OUT_DIR_DYN/engines/bf16_${TAG}.engine"
CRL_ONNX="$OUT_DIR_DYN/onnx/fp32_${TAG}_crl.onnx"
INT8_CRL_ONNX="$OUT_DIR_DYN/onnx/int8_${TAG}_crl.onnx"
CLEAN_CRL_ONNX="$OUT_DIR_DYN/onnx/int8_${TAG}_crl_clean.onnx"
INT8_CRL_ENG="$OUT_DIR_DYN/engines/int8_${TAG}_crl.engine"

# 1. BF16 dynamic engine (optional — useful for cos baseline)
if [[ "$SKIP_BF16" != "1" && ! -f "$BF16_ENG" ]]; then
    echo "=== 1. BF16 dynamic engine ==="
    "$PY" "$PE_INT8/build_dynamic_engine.py" \
        --onnx "$FP32_ONNX" --save_engine "$BF16_ENG" \
        --min_shape "$MIN_SHAPE" --opt_shape "$OPT_SHAPE" --max_shape "$MAX_SHAPE" \
        --mode bf16 \
        2>&1 | tee "$OUT_DIR_DYN/logs/build_bf16_dyn.log"
fi

# 2. CRL pre-pass on the FP32 ONNX (CRL needs the calibration sample shape,
#    not the engine opt shape, so we pass --batch_size $CALIB_BT here).
if [[ ! -f "$CRL_ONNX" ]]; then
    echo "=== 2. CRL pre-pass (σ_k=$SIGMA_K, calib at BT=$CALIB_BT) ==="
    "$PY" "$PE_INT8/crl_pass.py" \
        --in_onnx "$FP32_ONNX" --out_onnx "$CRL_ONNX" \
        --calib_npy "$CALIB_NPY" --sigma_k "$SIGMA_K" --batch_size "$CALIB_BT" \
        2>&1 | tee "$OUT_DIR_DYN/logs/crl_pass_dyn.log"
fi

# 3. modelopt PTQ on the CRL ONNX, calibrating at CALIB_BT (smaller than OPT
#    to dodge the ORT BFC 226 MB single-alloc cap at ViT-L MLP/attn layers).
#    Use 'entropy' for small BT (better quant scales), 'max' otherwise.
CAL_METHOD=$([ "$CALIB_BT" -lt 12 ] && echo "entropy" || ([ "$CALIB_BT" -le 12 ] && echo "entropy" || echo "max"))
if [[ ! -f "$INT8_CRL_ONNX" ]]; then
    echo "=== 3. modelopt PTQ on CRL+dynamic ONNX (calib BT=$CALIB_BT, method=$CAL_METHOD) ==="
    "$PY" "$PE_INT8/quantize_onnx.py" \
        --onnx_path "$CRL_ONNX" --output_path "$INT8_CRL_ONNX" \
        --img_size "$IMG_SIZE" --frames_per_video "$T_FRAMES" \
        --n_per_split 100 --seed 20260508 \
        --dataset_root "$DATASET_ROOT" \
        --manifest "$OUT_DIR_DYN/calib/manifest_val.json" \
        --calib_npy "$CALIB_NPY" --target_calib_n 96 \
        --calibration_shapes "$CALIB_SHAPE" \
        --disable_mha_qdq --high_precision_dtype fp16 \
        --calibration_method "$CAL_METHOD" --stratified --split val --exclude_patch_embed \
        --extra_args --calibration_eps cuda:0 cpu \
        2>&1 | tee "$OUT_DIR_DYN/logs/quantize_crl_dyn.log"
fi

# 4. Surgery (strip layout-only Q/DQ)
if [[ ! -f "$CLEAN_CRL_ONNX" ]]; then
    echo "=== 4. surgery on INT8+CRL ONNX ==="
    "$PY" "$PE_INT8/surgery.py" --onnx "$INT8_CRL_ONNX" --out "$CLEAN_CRL_ONNX" \
        2>&1 | tee "$OUT_DIR_DYN/logs/surgery_crl_dyn.log"
fi

# 5. Build INT8+CRL dynamic engine
echo "=== 5. build INT8+CRL dynamic engine ==="
"$PY" "$PE_INT8/build_dynamic_engine.py" \
    --onnx "$CLEAN_CRL_ONNX" --save_engine "$INT8_CRL_ENG" \
    --min_shape "$MIN_SHAPE" --opt_shape "$OPT_SHAPE" --max_shape "$MAX_SHAPE" \
    --mode strongly_typed --workspace_GiB 8.0 \
    2>&1 | tee "$OUT_DIR_DYN/logs/build_int8_crl_dyn.log"

echo
echo "=== DONE ==="
ls -lh "$OUT_DIR_DYN/engines/"
