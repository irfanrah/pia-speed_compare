#!/usr/bin/env bash
# run_on_a4000.sh — calibrate + bench a PE INT8 deploy on a fresh GPU (A4000/A6000/A100).
#
# This is a one-shot, self-contained runner. After copying the standalone
# claude_exp8_finish1/ tree, the pia-prompt_optimization PE vendor, and a
# QAT-trained deploy candidate (.pt), this script:
#   1. Exports FP32 ONNX from the .pt (trace at BT = BATCH * T_FRAMES)
#   2. Builds a TRT BF16 reference engine
#   3. Calibrates INT8 via modelopt-onnx PTQ on the val/ split
#   4. Optionally applies a CRL pre-pass + recalibrates -> INT8+CRL engine
#   5. Strips layout-only Q/DQ
#   6. Benches every engine against the test/ split, reporting cos vs PT BF16
#
# Usage:
#   PE_VENDOR=/path/to/pia-prompt_optimization \
#   PT_CKPT=/path/to/qat_deploy_fp32.pt \
#   DATASET_ROOT=/path/to/clips \
#   bash run_on_a4000.sh
#
# Env knobs (all optional):
#   T_FRAMES    3 for FT video (default), 1 for ZS image
#   BATCH       clips per inference (default 4)
#   SIGMA_K     CRL clip threshold (default 2.5 for FT, 3.0 for ZS)
#   OUT_DIR     where to write engines + bench results (default ./a4000_run)
#   GPU         CUDA_VISIBLE_DEVICES (default 0)
#   SKIP_CRL    set to 1 to build only BF16 + plain INT8 (faster)

set -euo pipefail

# Resolve paths
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"
PE_INT8="$EXP_ROOT/pe_int8"
[[ -d "$PE_INT8" ]] || { echo "ERR: bundle pe_int8/ not found at $PE_INT8" >&2; exit 1; }

# Required env
: "${PE_VENDOR:?set PE_VENDOR=/path/to/pia-prompt_optimization}"
: "${PT_CKPT:?set PT_CKPT=/path/to/qat_deploy_fp32.pt}"
: "${DATASET_ROOT:?set DATASET_ROOT=/path/to/clips (must contain val/ and test/)}"

[[ -d "$PE_VENDOR/src/PE/perception_models" ]] || { echo "ERR: PE_VENDOR layout wrong: $PE_VENDOR" >&2; exit 1; }
[[ -f "$PT_CKPT" ]] || { echo "ERR: PT_CKPT not found: $PT_CKPT" >&2; exit 1; }
[[ -d "$DATASET_ROOT/val" && -d "$DATASET_ROOT/test" ]] || { echo "ERR: $DATASET_ROOT must contain val/ and test/" >&2; exit 1; }

T_FRAMES="${T_FRAMES:-3}"
BATCH="${BATCH:-4}"
SIGMA_K="${SIGMA_K:-2.5}"
OUT_DIR="${OUT_DIR:-$(pwd)/a4000_run}"
GPU="${GPU:-0}"
SKIP_CRL="${SKIP_CRL:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PE_VENDOR
export PE_PMODELS="$PE_VENDOR/src/PE/perception_models"
export PYTHONPATH="$PE_INT8:$PE_PMODELS:$PE_VENDOR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$EXP_ROOT/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
PY="${PYTHONBIN:-$(command -v python3)}"

BT=$((BATCH * T_FRAMES))
TAG="b${BATCH}t${T_FRAMES}"
mkdir -p "$OUT_DIR"/{onnx,engines,calib,logs,results}

echo "[a4000]   PT_CKPT:      $PT_CKPT"
echo "[a4000]   PE_VENDOR:    $PE_VENDOR"
echo "[a4000]   pe_int8:      $PE_INT8"
echo "[a4000]   DATASET_ROOT: $DATASET_ROOT"
echo "[a4000]   T_FRAMES:     $T_FRAMES   BATCH: $BATCH   BT: $BT"
echo "[a4000]   OUT_DIR:      $OUT_DIR"
echo "[a4000]   GPU:          $GPU   SKIP_CRL: $SKIP_CRL"
echo

FP32="$OUT_DIR/onnx/fp32_${TAG}.onnx"
INT8="$OUT_DIR/onnx/int8_${TAG}.onnx"
INT8_CRL="$OUT_DIR/onnx/int8_${TAG}_crl.onnx"
CLEAN="$OUT_DIR/onnx/int8_${TAG}_clean.onnx"
CLEAN_CRL="$OUT_DIR/onnx/int8_${TAG}_crl_clean.onnx"
FP32_CRL="$OUT_DIR/onnx/fp32_${TAG}_crl.onnx"
ENG_BF16="$OUT_DIR/engines/bf16_${TAG}.engine"
ENG_INT8="$OUT_DIR/engines/int8_${TAG}.engine"
ENG_INT8_CRL="$OUT_DIR/engines/int8_${TAG}_crl.engine"
CALIB="$OUT_DIR/calib/calibration.npy"

CALIBRATION_METHOD=$([ "$BT" -lt 12 ] && echo "entropy" || echo "max")
SIMPLIFY_FLAG=$([ "$BT" -lt 12 ] && echo "--simplify" || echo "")

# ── 1. Export FP32 ONNX ────────────────────────────────────────────────────
echo "=== 1. export FP32 ONNX (trace BT=$BT) ==="
"$PY" "$PE_INT8/export_onnx.py" \
    --config_name PE-Core-L14-336 --ft_ckpt "$PT_CKPT" \
    --out_dir "$OUT_DIR/onnx" --out_name "fp32_${TAG}.onnx" \
    --batch_videos "$BATCH" --frames_per_video "$T_FRAMES" \
    2>&1 | tee "$OUT_DIR/logs/export.log"

# ── 2. BF16 reference engine ───────────────────────────────────────────────
echo "=== 2. build BF16 reference engine ==="
"$PY" "$PE_INT8/build_engine_py.py" \
    --onnx "$FP32" --save_engine "$ENG_BF16" \
    --shape "input:${BT}x3x336x336" --mode bf16 \
    --iters 30 --warmup 5 \
    2>&1 | tee "$OUT_DIR/logs/build_bf16.log"

# ── 3. INT8 PTQ from val/ ──────────────────────────────────────────────────
echo "=== 3. modelopt PTQ on val/ ==="
"$PY" "$PE_INT8/quantize_onnx.py" \
    --onnx_path "$FP32" --output_path "$INT8" \
    --img_size 336 --frames_per_video "$T_FRAMES" \
    --n_per_split 100 --seed 20260508 \
    --dataset_root "$DATASET_ROOT" \
    --manifest "$OUT_DIR/calib/manifest_val.json" \
    --calib_npy "$CALIB" --target_calib_n 96 \
    --calibration_shapes "input:${BT}x3x336x336" \
    --disable_mha_qdq --high_precision_dtype fp16 \
    --calibration_method "$CALIBRATION_METHOD" \
    --stratified --split val --exclude_patch_embed $SIMPLIFY_FLAG \
    --extra_args --calibration_eps cuda:0 cpu \
    2>&1 | tee "$OUT_DIR/logs/quantize.log"

# ── 4. Surgery + INT8 engine ───────────────────────────────────────────────
echo "=== 4. surgery + build INT8 engine ==="
"$PY" "$PE_INT8/surgery.py" --onnx "$INT8" --out "$CLEAN" \
    2>&1 | tee "$OUT_DIR/logs/surgery.log"
"$PY" "$PE_INT8/build_engine_py.py" \
    --onnx "$CLEAN" --save_engine "$ENG_INT8" \
    --shape "input:${BT}x3x336x336" --mode strongly_typed \
    --iters 30 --warmup 5 \
    2>&1 | tee "$OUT_DIR/logs/build_int8.log"

# ── 5. CRL + INT8+CRL engine ───────────────────────────────────────────────
if [[ "$SKIP_CRL" != "1" ]]; then
  echo "=== 5. CRL (σ_k=$SIGMA_K) → INT8+CRL engine ==="
  "$PY" "$PE_INT8/crl_pass.py" \
      --in_onnx "$FP32" --out_onnx "$FP32_CRL" \
      --calib_npy "$CALIB" --sigma_k "$SIGMA_K" --batch_size "$BT" \
      2>&1 | tee "$OUT_DIR/logs/crl_pass.log"
  "$PY" "$PE_INT8/quantize_onnx.py" \
      --onnx_path "$FP32_CRL" --output_path "$INT8_CRL" \
      --img_size 336 --frames_per_video "$T_FRAMES" \
      --n_per_split 100 --seed 20260508 \
      --dataset_root "$DATASET_ROOT" \
      --manifest "$OUT_DIR/calib/manifest_val.json" \
      --calib_npy "$CALIB" --target_calib_n 96 \
      --calibration_shapes "input:${BT}x3x336x336" \
      --disable_mha_qdq --high_precision_dtype fp16 \
      --calibration_method "$CALIBRATION_METHOD" \
      --stratified --split val --exclude_patch_embed $SIMPLIFY_FLAG \
      --extra_args --calibration_eps cuda:0 cpu \
      2>&1 | tee "$OUT_DIR/logs/quantize_crl.log"
  "$PY" "$PE_INT8/surgery.py" --onnx "$INT8_CRL" --out "$CLEAN_CRL" \
      2>&1 | tee "$OUT_DIR/logs/surgery_crl.log"
  "$PY" "$PE_INT8/build_engine_py.py" \
      --onnx "$CLEAN_CRL" --save_engine "$ENG_INT8_CRL" \
      --shape "input:${BT}x3x336x336" --mode strongly_typed \
      --iters 30 --warmup 5 \
      2>&1 | tee "$OUT_DIR/logs/build_int8_crl.log"
fi

# ── 6. Bench on test/ ──────────────────────────────────────────────────────
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

echo
echo "=== DONE ==="
echo "  results.json:  $OUT_DIR/results/results.json"
echo "  results.md:    $OUT_DIR/results/results.md"
echo
echo "Headline (cos / ms per engine):"
sed -n '/===== B=/,/verdict/p' "$OUT_DIR/results/results.md" 2>/dev/null || cat "$OUT_DIR/results/results.md"
