#!/usr/bin/env bash
# run_all.sh — sweep PE / FT_PE / PE_INT8 / FT_PE_INT8 speed benches
# across a range of batch sizes on a single GPU.
#
# Required:
#   GPU=<device-index>      e.g. GPU=0 — the CUDA device to pin to.
#
# Optional sweep knobs (defaults shown):
#   B_MIN=10                first BATCH in the sweep
#   B_MAX=70                last BATCH in the sweep
#   B_STEP=2                BATCH increment
#   FRAMES_FTPE=3           T (temporal frames) for FT_PE and FT_PE_INT8
#   DELAY_SEC=180           seconds to sleep BETWEEN iterations (per-B,
#                           after the four benches at that B have run).
#                           Set DELAY_SEC=0 to disable cool-down sleeps.
#   WARMUP=5  ITERS=25      forwarded to each speed bench
#   SKIP_COS=0              set to 1 to skip the cos/MSE pass on INT8 runs
#   BENCH_LIST=             space-separated subset of {PE FT_PE PE_INT8
#                           FT_PE_INT8}; defaults to all four
#
# Output:
#   results/<bench>_<gpu>_b<B>_..._<timestamp>.json   per-bench
#   results/sweep_logs/<bench>_b<B>.log               per-bench stdout
#   results/sweep_logs/run_all_<timestamp>.summary    one-line-per-bench
#                                                     pass/fail summary
#
# Bench order within each B (deliberate — least → most thermal load,
# so the GPU has a chance to warm up gradually):
#
#   1. PE          (BF16 zero-shot, T=1)
#   2. FT_PE       (BF16 fine-tuned, T=$FRAMES_FTPE)
#   3. PE_INT8     (INT8 zero-shot,   T=1)
#   4. FT_PE_INT8  (INT8 fine-tuned,  T=$FRAMES_FTPE)
#
# Each iteration writes four JSONs under results/. The INT8 wrappers
# additionally append a ``cos_mse`` section to their speed-bench JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Required: GPU ──────────────────────────────────────────────────────
if [ -z "${GPU:-}" ]; then
    echo "ERR: set GPU=<device-index>, e.g. GPU=0 bash scripts/run_all.sh" >&2
    echo "     (run nvidia-smi to see available device indices)" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"

# ── Sweep knobs ────────────────────────────────────────────────────────
B_MIN=${B_MIN:-10}
B_MAX=${B_MAX:-70}
B_STEP=${B_STEP:-2}
FRAMES_FTPE=${FRAMES_FTPE:-3}
DELAY_SEC=${DELAY_SEC:-180}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}
SKIP_COS=${SKIP_COS:-0}
BENCH_LIST=${BENCH_LIST:-"PE FT_PE PE_INT8 FT_PE_INT8"}

# Validate B_MIN <= B_MAX
if [ "$B_MIN" -gt "$B_MAX" ]; then
    echo "ERR: B_MIN ($B_MIN) > B_MAX ($B_MAX)" >&2; exit 2
fi

# ── Logging dir ────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/run_all_${TS}.summary"
: > "$SUMMARY"

GPU_NAME="$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"

echo "[sweep] GPU=$GPU ($GPU_NAME)"
echo "[sweep] B range: $B_MIN..$B_MAX step $B_STEP"
echo "[sweep] FRAMES_FTPE=$FRAMES_FTPE  WARMUP=$WARMUP  ITERS=$ITERS"
echo "[sweep] benches: $BENCH_LIST"
echo "[sweep] inter-iter delay: ${DELAY_SEC}s"
echo "[sweep] summary -> $SUMMARY"
echo

# ── Bench dispatch ─────────────────────────────────────────────────────
# run_one <bench-name> <B>  → echos "[bench] starting ..." then runs the
# wrapper, redirecting stdout+stderr to a per-bench log so the sweep
# console stays readable. Records a one-line summary regardless of
# success/failure.
run_one() {
    local bench="$1" B="$2"
    local log="$LOG_DIR/${bench}_b${B}.log"
    local t_start t_end rc
    t_start=$(date +%s)
    echo "[$(date +%H:%M:%S)] [sweep] run $bench  B=$B  -> $log"
    case "$bench" in
        PE)
            BATCH="$B" WARMUP="$WARMUP" ITERS="$ITERS" \
                bash scripts/speed_calculate_PE.sh \
                >"$log" 2>&1 && rc=0 || rc=$?
            ;;
        FT_PE)
            BATCH="$B" FRAMES="$FRAMES_FTPE" WARMUP="$WARMUP" ITERS="$ITERS" \
                bash scripts/speed_calculate_FTPE.sh \
                >"$log" 2>&1 && rc=0 || rc=$?
            ;;
        PE_INT8)
            BATCH="$B" WARMUP="$WARMUP" ITERS="$ITERS" SKIP_COS="$SKIP_COS" \
                bash scripts/speed_calculate_PE_INT8.sh \
                >"$log" 2>&1 && rc=0 || rc=$?
            ;;
        FT_PE_INT8)
            BATCH="$B" FRAMES="$FRAMES_FTPE" WARMUP="$WARMUP" ITERS="$ITERS" SKIP_COS="$SKIP_COS" \
                bash scripts/speed_calculate_FTPE_INT8.sh \
                >"$log" 2>&1 && rc=0 || rc=$?
            ;;
        *)
            echo "[sweep] WARN: unknown bench '$bench'; skipping" >&2
            return 0
            ;;
    esac
    t_end=$(date +%s)
    local dt=$((t_end - t_start))
    if [ "$rc" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] [sweep] OK   $bench  B=$B  (${dt}s)"
        printf '%s B=%-3d %-12s OK    %4ds\n' "$(date +%H:%M:%S)" "$B" "$bench" "$dt" >>"$SUMMARY"
    else
        echo "[$(date +%H:%M:%S)] [sweep] FAIL $bench  B=$B  (rc=$rc, ${dt}s)  see $log" >&2
        printf '%s B=%-3d %-12s FAIL  %4ds  rc=%d\n' "$(date +%H:%M:%S)" "$B" "$bench" "$dt" "$rc" >>"$SUMMARY"
    fi
}

# ── Main sweep ─────────────────────────────────────────────────────────
first_iter=1
for B in $(seq "$B_MIN" "$B_STEP" "$B_MAX"); do
    if [ "$first_iter" -eq 0 ] && [ "$DELAY_SEC" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] [sweep] sleeping ${DELAY_SEC}s before next B=$B"
        sleep "$DELAY_SEC"
    fi
    first_iter=0
    echo "[$(date +%H:%M:%S)] [sweep] === iteration B=$B ==="
    for bench in $BENCH_LIST; do
        run_one "$bench" "$B"
    done
done

echo
echo "[sweep] done. summary:"
cat "$SUMMARY"
