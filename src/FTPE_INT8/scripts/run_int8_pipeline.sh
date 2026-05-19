#!/usr/bin/env bash
# run_int8_pipeline.sh — build + calibrate + bench an INT8 TRT engine for
# zero-shot PE or fine-tuned FT_PE on any CUDA GPU.
#
# Two model variants (VARIANT env knob):
#
#   VARIANT=pe   (default)  — zero-shot PE, T=1, STATIC engine at BATCH=16.
#                Default PT_CKPT auto-downloads from
#                  huggingface.co/PIA-SPACE-LAB/PE-Core-L14-336
#                    /resolve/main/splitqkv_qat/qat_deploy_fp32.pt
#
#   VARIANT=ftpe            — fine-tuned PE, T=3, DYNAMIC engine with
#                profile B_MIN..B_MAX (default 4..128, opt at 16).
#                PTQ runs at B_CALIB=4 (BT=12) to stay under the ORT BFC
#                arena 226 MB single-alloc cap on ViT-L MLP/attn layers.
#                Default PT_CKPT auto-downloads from
#                  huggingface.co/PIA-SPACE-LAB/FT_PE-Core-L14-336_260318
#                    /resolve/main/splitqkv_qat_t3/qat_deploy_fp32.pt
#
# Calibration + bench clips auto-download from
#   huggingface.co/datasets/PIA-SPACE-LAB/PE_INT8_QAT_CRL
# when DATASET_ROOT is unset.
#
# ────────────────────────────────────────────────────────────────────────
# Three input modes, picked by what you set:
#
#   (A) Full pipeline (PT → ONNX → INT8 engine):
#         VARIANT=pe \                # or ftpe
#         bash run_int8_pipeline.sh
#         # PT_CKPT + DATASET_ROOT auto-fetch from HF if unset.
#
#   (B) Skip the PT→ONNX export by reusing a pre-built FP32 ONNX:
#         FP32_ONNX=/path/to/fp32_<tag>.onnx \
#         CALIB_NPY=/path/to/calibration.npy \   # optional
#         bash run_int8_pipeline.sh
#
#   (C) Engine-only build from a pre-quantized INT8 ONNX:
#         CLEAN_INT8_ONNX=/path/to/int8_<tag>_clean.onnx \
#         SKIP_BENCH=1 \
#         bash run_int8_pipeline.sh
#
# Pipeline stages (each step is auto-skipped if its output already exists):
#   1. Export FP32 ONNX from the .pt        (requires PT_CKPT + PE_VENDOR)
#   2. Build TRT BF16 reference engine
#   3. modelopt-onnx PTQ on the val/ split  (requires DATASET_ROOT or CALIB_NPY)
#   4. ONNX surgery (strip layout-only Q/DQ) → INT8 engine
#   5. Optional CRL pre-pass + recalibrate → INT8+CRL engine
#   6. Bench every engine against test/, cos vs PT BF16    (skip via SKIP_BENCH=1)
#
# Common env knobs:
#   VARIANT     pe (default) | ftpe
#   BATCH       static-engine BT base (default: pe → 16; ignored when dynamic)
#   T_FRAMES    auto-set per VARIANT (pe → 1, ftpe → 3)
#   SIGMA_K     CRL clip threshold (default: pe → 3.0, ftpe → 2.5)
#   OUT_DIR     where to write engines + bench results (default ./int8_run_<variant>)
#   GPU         CUDA_VISIBLE_DEVICES (default 0)
#   SKIP_CRL    1 to build only BF16 + plain INT8 (faster)
#   SKIP_BENCH  1 to skip the cos-vs-PT-BF16 evaluation
#
# Dynamic-engine knobs (used when VARIANT=ftpe or BUILD_MODE=dynamic):
#   B_MIN B_OPT B_MAX   engine profile (default 4 / 16 / 128)
#   B_CALIB             PTQ batch (default 4 — must satisfy
#                         B_CALIB*T <= ORT BFC arena cap)
#   BUILD_MODE          static | dynamic (default per VARIANT)
#
# Skip-mode overrides:
#   PT_CKPT            override the auto-fetched checkpoint
#   PE_VENDOR          path to pia-prompt_optimization (auto-resolved if
#                      unset and present at <pe_int8>/vendor/...)
#   DATASET_ROOT       override the auto-fetched dataset
#   FP32_ONNX          pre-built FP32 ONNX → skips step 1
#   CALIB_NPY          pre-built calibration tensor → skips PTQ calibration
#   CLEAN_INT8_ONNX    pre-built clean INT8 ONNX → skips steps 1+3+4

set -euo pipefail

# ── Variant defaults ─────────────────────────────────────────────────────
VARIANT="${VARIANT:-pe}"
case "$VARIANT" in
    pe)
        : "${T_FRAMES:=1}"
        : "${BATCH:=16}"
        : "${BUILD_MODE:=static}"
        : "${SIGMA_K:=3.0}"
        _PT_HF_REPO="PIA-SPACE-LAB/PE-Core-L14-336"
        _PT_HF_FILE="splitqkv_qat/qat_deploy_fp32.pt"
        ;;
    ftpe)
        : "${T_FRAMES:=3}"
        : "${BUILD_MODE:=dynamic}"
        : "${SIGMA_K:=2.5}"
        : "${B_MIN:=4}"
        : "${B_OPT:=16}"
        : "${B_MAX:=128}"
        : "${B_CALIB:=4}"
        : "${BATCH:=$B_CALIB}"   # used for static-fallback TAG; ignored when dynamic
        _PT_HF_REPO="PIA-SPACE-LAB/FT_PE-Core-L14-336_260318"
        _PT_HF_FILE="splitqkv_qat_t3/qat_deploy_fp32.pt"
        ;;
    *)
        echo "ERR: unknown VARIANT=$VARIANT (use 'pe' or 'ftpe')" >&2; exit 1
        ;;
esac

case "$BUILD_MODE" in static|dynamic) ;;
    *) echo "ERR: BUILD_MODE must be static or dynamic, got $BUILD_MODE" >&2; exit 1;;
esac

OUT_DIR="${OUT_DIR:-$(pwd)/int8_run_${VARIANT}}"
GPU="${GPU:-0}"
SKIP_CRL="${SKIP_CRL:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"

# Resolve paths
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
PE_INT8="$EXP_ROOT/pe_int8"
[[ -d "$PE_INT8" ]] || { echo "ERR: bundle pe_int8/ not found at $PE_INT8" >&2; exit 1; }

# ── HF cache ─────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$EXP_ROOT/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME"

# Lift the cached login token from the user's default HF_HOME so private
# repos work even when we point HF_HOME at a project-local cache. (The
# library only reads $HF_HOME/token; without this it sees no token and
# 401s on private models.)
if [[ -z "${HF_TOKEN:-}" ]]; then
    for _tok in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
        if [[ -f "$_tok" ]]; then
            export HF_TOKEN="$(<"$_tok")"
            break
        fi
    done
fi

# ── HF download helpers ──────────────────────────────────────────────────
# Use the Python API (huggingface_hub) directly so the cached token at
# ~/.cache/huggingface/token is picked up automatically — private repos
# fail under `huggingface-cli download` on some setups even when whoami works.
_PY_FOR_HF="${PYTHONBIN:-$(command -v python3)}"

_hf_download_file() {
    local repo="$1" path_in_repo="$2" out_root="$3"
    local dest="$out_root/$path_in_repo"
    if [[ -f "$dest" ]]; then echo "$dest"; return 0; fi
    mkdir -p "$out_root" "$(dirname "$dest")"
    "$_PY_FOR_HF" -c '
import sys
from huggingface_hub import hf_hub_download
repo, path_in_repo, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"[hf] hf_hub_download({repo!r}, {path_in_repo!r}) -> {out_root}", flush=True)
p = hf_hub_download(repo_id=repo, filename=path_in_repo,
                    repo_type="model", local_dir=out_root)
print(f"[hf] downloaded: {p}", flush=True)
' "$repo" "$path_in_repo" "$out_root" >&2 || {
        echo "ERR: HF download failed: $repo/$path_in_repo" >&2; exit 1; }
    [[ -f "$dest" ]] || { echo "ERR: HF download produced no file at $dest" >&2; exit 1; }
    echo "$dest"
}

_hf_download_dataset() {
    local repo="$1" out_dir="$2"
    if [[ -d "$out_dir" && -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
        echo "$out_dir"; return 0
    fi
    mkdir -p "$out_dir"
    "$_PY_FOR_HF" -c '
import sys
from huggingface_hub import snapshot_download
repo, out_dir = sys.argv[1], sys.argv[2]
print(f"[hf] snapshot_download({repo!r}, repo_type=\"dataset\") -> {out_dir}", flush=True)
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out_dir)
print(f"[hf] done -> {out_dir}", flush=True)
' "$repo" "$out_dir" >&2 || {
        echo "ERR: HF dataset download failed: $repo" >&2; exit 1; }
    echo "$out_dir"
}

# ── Conditional preflight ────────────────────────────────────────────────
# (C) clean INT8 ONNX provided → no PT/vendor/dataset needed at all.
# (B) FP32 ONNX provided       → no PT/vendor; dataset only if CALIB_NPY missing.
# (A) full pipeline            → PT + vendor + dataset all required.
need_pt_for_export=1
need_dataset_for_calib=1
if [[ -n "${CLEAN_INT8_ONNX:-}" ]]; then
    need_pt_for_export=0
    need_dataset_for_calib=0
elif [[ -n "${FP32_ONNX:-}" ]]; then
    need_pt_for_export=0
    [[ -n "${CALIB_NPY:-}" && -f "${CALIB_NPY}" ]] && need_dataset_for_calib=0
fi

# Auto-resolve PE_VENDOR
if [[ "$need_pt_for_export" -eq 1 && -z "${PE_VENDOR:-}" ]]; then
    for c in "$EXP_ROOT/vendor/pia-prompt_optimization" \
             "$EXP_ROOT/../../pia-prompt_optimization" \
             "$EXP_ROOT/../../../pia-prompt_optimization"; do
        if [[ -d "$c/src/PE/perception_models" ]]; then
            PE_VENDOR="$(cd "$c" && pwd)"
            break
        fi
    done
fi
if [[ "$need_pt_for_export" -eq 1 ]]; then
    [[ -n "${PE_VENDOR:-}" && -d "$PE_VENDOR/src/PE/perception_models" ]] || \
        { echo "ERR: PE_VENDOR not found. Set PE_VENDOR=/path/to/pia-prompt_optimization (or set FP32_ONNX/CLEAN_INT8_ONNX to skip export)." >&2; exit 1; }
fi

# Auto-fetch PT_CKPT if we still need export and the variant has an HF default.
if [[ "$need_pt_for_export" -eq 1 && -z "${PT_CKPT:-}" ]]; then
    if [[ -n "$_PT_HF_REPO" ]]; then
        echo "[hf] fetching PT checkpoint: $_PT_HF_REPO/$_PT_HF_FILE"
        PT_CKPT="$(_hf_download_file "$_PT_HF_REPO" "$_PT_HF_FILE" "$HF_HOME/$VARIANT")"
    else
        echo "ERR: VARIANT=$VARIANT has no default HF PT_CKPT. Set PT_CKPT=/path/to/qat_deploy.pt (or FP32_ONNX=/path/to/fp32.onnx to skip export)." >&2; exit 1
    fi
fi
if [[ "$need_pt_for_export" -eq 1 ]]; then
    [[ -f "$PT_CKPT" ]] || { echo "ERR: PT_CKPT not found: $PT_CKPT" >&2; exit 1; }
fi

# Auto-fetch DATASET_ROOT if calibration or bench still needs clips.
if [[ ( "$need_dataset_for_calib" -eq 1 || "$SKIP_BENCH" != "1" ) && -z "${DATASET_ROOT:-}" ]]; then
    echo "[hf] fetching CRL/cos dataset: PIA-SPACE-LAB/PE_INT8_QAT_CRL"
    DATASET_ROOT="$(_hf_download_dataset "PIA-SPACE-LAB/PE_INT8_QAT_CRL" "$HF_HOME/PE_INT8_QAT_CRL")"
fi

if [[ "$need_dataset_for_calib" -eq 1 ]]; then
    [[ -d "$DATASET_ROOT/val" ]] || { echo "ERR: $DATASET_ROOT must contain val/ (or set CALIB_NPY to a pre-built calib tensor)" >&2; exit 1; }
fi
if [[ "$SKIP_BENCH" != "1" ]]; then
    [[ -d "$DATASET_ROOT/test" ]] || { echo "ERR: $DATASET_ROOT must contain test/ (or set SKIP_BENCH=1)" >&2; exit 1; }
fi

# ── Environment ──────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPU"
if [[ -n "${PE_VENDOR:-}" ]]; then
    export PE_VENDOR
    export PE_PMODELS="$PE_VENDOR/src/PE/perception_models"
    export PYTHONPATH="$PE_INT8:$PE_PMODELS:$PE_VENDOR${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$PE_INT8${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONUNBUFFERED=1
PY="${PYTHONBIN:-$(command -v python3)}"

# Make CUDA-bound libs from the pip-installed nvidia-* wheels visible to ORT.
# onnxruntime-gpu doesn't ship its own cuDNN; without this on LD_LIBRARY_PATH
# the CUDAExecutionProvider fails to load (libcudnn.so.9 missing) and PTQ
# silently falls back to CPU -- OK at BT=12, OOMs the BFC arena at BT>=48.
_NV_LIBS="$("$PY" -c '
import os, importlib, site
libs=[]
# Some nvidia-* wheels expose a real submodule (have __file__); others are
# namespace packages with __file__=None. For the namespace-package case,
# fall back to <site-packages>/nvidia/<name>/lib.
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
    except Exception:
        pass
    if not p or not os.path.isdir(p):
        for r in roots:
            cand = os.path.join(r, name, "lib")
            if os.path.isdir(cand): p = cand; break
    if p and os.path.isdir(p) and p not in libs:
        libs.append(p)
print(":".join(libs))
' 2>/dev/null || true)"
if [[ -n "$_NV_LIBS" ]]; then
    export LD_LIBRARY_PATH="$_NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ── Shapes / tags ────────────────────────────────────────────────────────
if [[ "$BUILD_MODE" == "static" ]]; then
    BT=$((BATCH * T_FRAMES))
    BT_TRACE="$BT"
    BT_CALIB="$BT"
    TAG="${VARIANT}_b${BATCH}t${T_FRAMES}"
else
    BT_MIN=$((B_MIN * T_FRAMES))
    BT_OPT=$((B_OPT * T_FRAMES))
    BT_MAX=$((B_MAX * T_FRAMES))
    BT_CALIB=$((B_CALIB * T_FRAMES))
    BT_TRACE="$BT_OPT"        # trace at the opt shape (dynamic_axes makes batch dim symbolic)
    BT="$BT_CALIB"             # used by the bench step (it runs at a concrete shape)
    TAG="${VARIANT}_dyn_b${B_MIN}-${B_MAX}_t${T_FRAMES}"
fi

mkdir -p "$OUT_DIR"/{onnx,engines,calib,logs,results}

GPU_NAME="$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
GPU_SM="$(nvidia-smi --id="$GPU" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || echo "?")"

echo "[int8]    VARIANT:      $VARIANT       BUILD_MODE: $BUILD_MODE"
echo "[int8]    GPU:          $GPU  ($GPU_NAME, sm $GPU_SM)"
echo "[int8]    pe_int8:      $PE_INT8"
echo "[int8]    PT_CKPT:      ${PT_CKPT:-<none — using pre-built ONNX>}"
echo "[int8]    PE_VENDOR:    ${PE_VENDOR:-<none>}"
echo "[int8]    DATASET_ROOT: ${DATASET_ROOT:-<none>}"
echo "[int8]    T_FRAMES:     $T_FRAMES"
if [[ "$BUILD_MODE" == "static" ]]; then
    echo "[int8]    BATCH:        $BATCH    BT: $BT"
else
    echo "[int8]    B_MIN/OPT/MAX: $B_MIN / $B_OPT / $B_MAX  (BT $BT_MIN..$BT_MAX, opt $BT_OPT)"
    echo "[int8]    B_CALIB:      $B_CALIB  (PTQ BT=$BT_CALIB)"
fi
echo "[int8]    OUT_DIR:      $OUT_DIR"
echo "[int8]    SKIP_CRL:     $SKIP_CRL     SKIP_BENCH: $SKIP_BENCH"
echo

# Artifact paths. FP32_ONNX / CLEAN_INT8_ONNX / CALIB_NPY env overrides let
# the user point at pre-built artifacts so the relevant stage gets skipped.
FP32="${FP32_ONNX:-$OUT_DIR/onnx/fp32_${TAG}.onnx}"
INT8="$OUT_DIR/onnx/int8_${TAG}.onnx"
INT8_CRL="$OUT_DIR/onnx/int8_${TAG}_crl.onnx"
CLEAN="${CLEAN_INT8_ONNX:-$OUT_DIR/onnx/int8_${TAG}_clean.onnx}"
CLEAN_CRL="$OUT_DIR/onnx/int8_${TAG}_crl_clean.onnx"
FP32_CRL="$OUT_DIR/onnx/fp32_${TAG}_crl.onnx"
ENG_BF16="$OUT_DIR/engines/bf16_${TAG}.engine"
ENG_INT8="$OUT_DIR/engines/int8_${TAG}.engine"
ENG_INT8_CRL="$OUT_DIR/engines/int8_${TAG}_crl.engine"
CALIB="${CALIB_NPY:-$OUT_DIR/calib/calibration.npy}"

CALIB_SHAPE="input:${BT_CALIB}x3x336x336"
CALIBRATION_METHOD=$([ "$BT_CALIB" -lt 12 ] && echo "entropy" || echo "max")
SIMPLIFY_FLAG=$([ "$BT_CALIB" -lt 12 ] && echo "--simplify" || echo "")

# Helper: build a TRT engine (static single-shape vs dynamic min/opt/max).
_build_engine() {
    local onnx="$1" out="$2" mode="$3" log="$4"
    if [[ "$BUILD_MODE" == "static" ]]; then
        "$PY" "$PE_INT8/build_engine_py.py" \
            --onnx "$onnx" --save_engine "$out" \
            --shape "input:${BT}x3x336x336" --mode "$mode" \
            --iters 30 --warmup 5 \
            2>&1 | tee "$log"
    else
        "$PY" "$PE_INT8/build_dynamic_engine.py" \
            --onnx "$onnx" --save_engine "$out" \
            --min_shape "input:${BT_MIN}x3x336x336" \
            --opt_shape "input:${BT_OPT}x3x336x336" \
            --max_shape "input:${BT_MAX}x3x336x336" \
            --mode "$mode" --workspace_GiB 8.0 \
            2>&1 | tee "$log"
    fi
}

# ── 1. Export FP32 ONNX ────────────────────────────────────────────────────
if [[ -f "$FP32" ]]; then
    echo "=== 1. FP32 ONNX present — skip export ($FP32) ==="
else
    echo "=== 1. export FP32 ONNX (trace BT=$BT_TRACE) ==="
    "$PY" "$PE_INT8/export_onnx.py" \
        --config_name PE-Core-L14-336 --ft_ckpt "$PT_CKPT" \
        --out_dir "$OUT_DIR/onnx" --out_name "fp32_${TAG}.onnx" \
        --batch_videos $((BT_TRACE / T_FRAMES)) --frames_per_video "$T_FRAMES" \
        2>&1 | tee "$OUT_DIR/logs/export.log"
fi

# ── 2. BF16 reference engine ───────────────────────────────────────────────
if [[ -f "$ENG_BF16" ]]; then
    echo "=== 2. BF16 engine present — skip build ($ENG_BF16) ==="
else
    echo "=== 2. build BF16 reference engine ($BUILD_MODE) ==="
    _build_engine "$FP32" "$ENG_BF16" bf16 "$OUT_DIR/logs/build_bf16.log"
fi

# ── 3+4. INT8 PTQ + surgery ────────────────────────────────────────────────
if [[ -f "$CLEAN" ]]; then
    echo "=== 3+4. clean INT8 ONNX present — skip PTQ + surgery ($CLEAN) ==="
else
    echo "=== 3. modelopt PTQ on val/ (calib BT=$BT_CALIB) ==="
    "$PY" "$PE_INT8/quantize_onnx.py" \
        --onnx_path "$FP32" --output_path "$INT8" \
        --img_size 336 --frames_per_video "$T_FRAMES" \
        --n_per_split 100 --seed 20260508 \
        --dataset_root "${DATASET_ROOT:-/dev/null}" \
        --manifest "$OUT_DIR/calib/manifest_val.json" \
        --calib_npy "$CALIB" --target_calib_n 96 \
        --calibration_shapes "$CALIB_SHAPE" \
        --disable_mha_qdq --high_precision_dtype fp16 \
        --calibration_method "$CALIBRATION_METHOD" \
        --stratified --split val --exclude_patch_embed $SIMPLIFY_FLAG \
        --extra_args --calibration_eps cuda:0 cpu \
        2>&1 | tee "$OUT_DIR/logs/quantize.log"
    echo "=== 4. surgery on INT8 ONNX ==="
    "$PY" "$PE_INT8/surgery.py" --onnx "$INT8" --out "$CLEAN" \
        2>&1 | tee "$OUT_DIR/logs/surgery.log"
fi

# Build the INT8 engine from the clean ONNX.
if [[ -f "$ENG_INT8" ]]; then
    echo "=== 4b. INT8 engine present — skip build ($ENG_INT8) ==="
else
    echo "=== 4b. build INT8 engine ($BUILD_MODE) ==="
    _build_engine "$CLEAN" "$ENG_INT8" strongly_typed "$OUT_DIR/logs/build_int8.log"
fi

# ── 5. CRL + INT8+CRL engine ───────────────────────────────────────────────
if [[ "$SKIP_CRL" != "1" ]]; then
  if [[ ! -f "$FP32_CRL" ]]; then
      echo "=== 5a. CRL pre-pass (σ_k=$SIGMA_K, calib BT=$BT_CALIB) ==="
      "$PY" "$PE_INT8/crl_pass.py" \
          --in_onnx "$FP32" --out_onnx "$FP32_CRL" \
          --calib_npy "$CALIB" --sigma_k "$SIGMA_K" --batch_size "$BT_CALIB" \
          2>&1 | tee "$OUT_DIR/logs/crl_pass.log"
  fi
  if [[ ! -f "$CLEAN_CRL" ]]; then
      echo "=== 5b. PTQ + surgery on CRL ONNX ==="
      "$PY" "$PE_INT8/quantize_onnx.py" \
          --onnx_path "$FP32_CRL" --output_path "$INT8_CRL" \
          --img_size 336 --frames_per_video "$T_FRAMES" \
          --n_per_split 100 --seed 20260508 \
          --dataset_root "${DATASET_ROOT:-/dev/null}" \
          --manifest "$OUT_DIR/calib/manifest_val.json" \
          --calib_npy "$CALIB" --target_calib_n 96 \
          --calibration_shapes "$CALIB_SHAPE" \
          --disable_mha_qdq --high_precision_dtype fp16 \
          --calibration_method "$CALIBRATION_METHOD" \
          --stratified --split val --exclude_patch_embed $SIMPLIFY_FLAG \
          --extra_args --calibration_eps cuda:0 cpu \
          2>&1 | tee "$OUT_DIR/logs/quantize_crl.log"
      "$PY" "$PE_INT8/surgery.py" --onnx "$INT8_CRL" --out "$CLEAN_CRL" \
          2>&1 | tee "$OUT_DIR/logs/surgery_crl.log"
  fi
  if [[ ! -f "$ENG_INT8_CRL" ]]; then
      echo "=== 5c. build INT8+CRL engine ($BUILD_MODE) ==="
      _build_engine "$CLEAN_CRL" "$ENG_INT8_CRL" strongly_typed "$OUT_DIR/logs/build_int8_crl.log"
  fi
fi

# ── 6. Bench on test/ (needs PT_CKPT for cos vs PT BF16) ───────────────────
if [[ "$SKIP_BENCH" == "1" ]]; then
    echo "=== 6. bench skipped (SKIP_BENCH=1) ==="
elif [[ -z "${PT_CKPT:-}" ]]; then
    echo "=== 6. bench skipped — PT_CKPT not provided (cos vs PT BF16 needs a .pt) ==="
else
    echo "=== 6. bench engines on test/ ==="
    "$PY" "$PE_INT8/bench_trt.py" \
        --config_name PE-Core-L14-336 --ft_ckpt "$PT_CKPT" \
        --frames_per_video "$T_FRAMES" \
        --engine_dir "$OUT_DIR/engines" \
        --manifest "$OUT_DIR/calib/manifest_test.json" \
        --dataset_root "$DATASET_ROOT" \
        --n_per_split 100 --seed 20260508 \
        --iters 30 --warmup 5 \
        --out_dir "$OUT_DIR/results" \
        --stratified --split test \
        2>&1 | tee "$OUT_DIR/logs/bench.log"
fi

echo
echo "=== DONE ==="
echo "  engines: $OUT_DIR/engines/"
if [[ -f "$OUT_DIR/results/results.md" ]]; then
    echo "  results.json:  $OUT_DIR/results/results.json"
    echo "  results.md:    $OUT_DIR/results/results.md"
    echo
    echo "Headline (cos / ms per engine):"
    sed -n '/===== B=/,/verdict/p' "$OUT_DIR/results/results.md" 2>/dev/null || cat "$OUT_DIR/results/results.md"
fi
