#!/usr/bin/env bash
# Sweep PE and FT_PE benchmarks across the full B / T grid with cooldowns
# between every run, so thermal throttling on the A4000 doesn't poison the
# tail of each run or carry over to the next.
#
# Order:
#   PE     : B = 4, 8, 10, 12, 14, 16
#   FT_PE  : T = 1, 3, 8 × B = 4, 8, 10, 12, 14, 16
# Total: 6 + 18 = 24 runs.
#
# Cooldowns:
#   INITIAL_DELAY  before the first run                          (default 10 min)
#   COOLDOWN       after every run, before the next              (default 10 min)
# The cooldown is also applied AFTER the last run so the GPU is left cool.
#
# Override via env vars (defaults in parens):
#   INITIAL_DELAY=600  (seconds)
#   COOLDOWN=600       (seconds)
#   WARMUP=5           (forwarded to the underlying bench scripts)
#   ITERS=25           (forwarded)
#   BATCHES="4 8 10 12 14 16"
#   FT_T_VALUES="1 3 8"
#   SKIP_PE=0          (set to 1 to skip the PE sweep)
#   SKIP_FTPE=0        (set to 1 to skip the FT_PE sweep)
#   DRY_RUN=0          (set to 1 to print the plan without running anything)
#   TAG=""             (suffix appended to every result file via --tag)
#
# Logs:
#   logs/run_all.<timestamp>.log  receives stdout+stderr of every run, with
#   GPU temperature / utilization snapshots taken before and after each run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INITIAL_DELAY=${INITIAL_DELAY:-600}
COOLDOWN=${COOLDOWN:-600}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-25}
BATCHES=${BATCHES:-"4 8 10 12 14 16"}
FT_T_VALUES=${FT_T_VALUES:-"1 3 8"}
SKIP_PE=${SKIP_PE:-0}
SKIP_FTPE=${SKIP_FTPE:-0}
DRY_RUN=${DRY_RUN:-0}
TAG=${TAG:-""}

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_all.${TS}.log"

# ----------------------------------------------------------------------------
# Helpers

human_secs() {
    # Format seconds as Hh Mm Ss for the banner.
    local s=$1
    printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

gpu_snapshot() {
    # One-line GPU stat snapshot, best-effort. Doesn't fail the run.
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null | head -1 \
            | awk -F', ' '{printf "gpu=%s  name=%s  temp=%s°C  util=%s%%  mem=%s/%s MiB\n",$1,$2,$3,$4,$5,$6}'
    else
        echo "(nvidia-smi unavailable)"
    fi
}

banner() {
    local msg="$1"
    {
        echo
        echo "════════════════════════════════════════════════════════════════════════════"
        echo " $msg"
        echo " $(date --iso-8601=seconds)    $(gpu_snapshot)"
        echo "════════════════════════════════════════════════════════════════════════════"
    } | tee -a "$LOG"
}

cooldown() {
    local secs=$1
    local label=$2
    if [ "$secs" -le 0 ]; then return; fi
    echo "[cooldown] $label: sleeping $(human_secs "$secs")  (until $(date -d "+$secs seconds" --iso-8601=seconds))" | tee -a "$LOG"
    if [ "$DRY_RUN" = "1" ]; then return; fi
    # Sleep in 60s chunks so a Ctrl-C lands within a second.
    local left=$secs
    while [ "$left" -gt 0 ]; do
        local step=60
        [ "$left" -lt "$step" ] && step=$left
        sleep "$step"
        left=$((left - step))
    done
    echo "[cooldown] done.  $(gpu_snapshot)" | tee -a "$LOG"
}

run_pe() {
    local b=$1
    banner "PE  | B=${b}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] BATCH=$b WARMUP=$WARMUP ITERS=$ITERS ./scripts/speed_calculate_PE.sh ${TAG:+-- --tag $TAG}" | tee -a "$LOG"
        return
    fi
    BATCH=$b WARMUP=$WARMUP ITERS=$ITERS \
        ./scripts/speed_calculate_PE.sh ${TAG:+-- --tag "$TAG"} 2>&1 | tee -a "$LOG"
}

run_ftpe() {
    local b=$1
    local t=$2
    banner "FT_PE | B=${b}  T=${t}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] BATCH=$b FRAMES=$t WARMUP=$WARMUP ITERS=$ITERS ./scripts/speed_calculate_FTPE.sh ${TAG:+-- --tag $TAG}" | tee -a "$LOG"
        return
    fi
    BATCH=$b FRAMES=$t WARMUP=$WARMUP ITERS=$ITERS \
        ./scripts/speed_calculate_FTPE.sh ${TAG:+-- --tag "$TAG"} 2>&1 | tee -a "$LOG"
}

# ----------------------------------------------------------------------------
# Plan

# shellcheck disable=SC2206
B_ARR=($BATCHES)
# shellcheck disable=SC2206
T_ARR=($FT_T_VALUES)

PE_RUNS=0
FTPE_RUNS=0
[ "$SKIP_PE" = "0" ]   && PE_RUNS=${#B_ARR[@]}
[ "$SKIP_FTPE" = "0" ] && FTPE_RUNS=$(( ${#B_ARR[@]} * ${#T_ARR[@]} ))
TOTAL_RUNS=$((PE_RUNS + FTPE_RUNS))

# Total wall time estimate: each run ~1-2 min on A4000, plus cooldowns.
# (TOTAL_RUNS - 1) gaps between runs, plus the initial delay and a final cooldown.
EST_COOLDOWN_S=$((INITIAL_DELAY + TOTAL_RUNS * COOLDOWN))
EST_BENCH_S=$((TOTAL_RUNS * 120))  # rough 2-min upper bound per run
EST_TOTAL_S=$((EST_COOLDOWN_S + EST_BENCH_S))

{
    echo "═════════ run_all sweep ═════════"
    echo "  start         : $(date --iso-8601=seconds)"
    echo "  log           : $LOG"
    echo "  PE batches    : ${BATCHES}                       (count=${#B_ARR[@]}, skip=$SKIP_PE)"
    echo "  FT_PE T values: ${FT_T_VALUES}                       (count=${#T_ARR[@]}, skip=$SKIP_FTPE)"
    echo "  WARMUP / ITERS: $WARMUP / $ITERS"
    echo "  INITIAL_DELAY : $(human_secs "$INITIAL_DELAY")"
    echo "  COOLDOWN      : $(human_secs "$COOLDOWN")  (also after the last run)"
    echo "  total runs    : $TOTAL_RUNS  (PE=$PE_RUNS, FT_PE=$FTPE_RUNS)"
    echo "  est. wall time: ~$(human_secs "$EST_TOTAL_S")  (cooldowns ~$(human_secs "$EST_COOLDOWN_S") + bench ~$(human_secs "$EST_BENCH_S"))"
    [ "$DRY_RUN" = "1" ] && echo "  DRY_RUN       : YES (no commands executed)"
    echo "═════════════════════════════════"
} | tee -a "$LOG"

if [ "$TOTAL_RUNS" -eq 0 ]; then
    echo "Nothing to run (both SKIP_PE and SKIP_FTPE are set)." | tee -a "$LOG"
    exit 0
fi

# ----------------------------------------------------------------------------
# Run

cooldown "$INITIAL_DELAY" "initial GPU cool-down"

i=0
if [ "$SKIP_PE" = "0" ]; then
    for b in "${B_ARR[@]}"; do
        i=$((i + 1))
        echo "[$i/$TOTAL_RUNS] starting PE run" | tee -a "$LOG"
        run_pe "$b"
        # Cooldown after every run, including the last one.
        cooldown "$COOLDOWN" "after run $i/$TOTAL_RUNS"
    done
fi

if [ "$SKIP_FTPE" = "0" ]; then
    for t in "${T_ARR[@]}"; do
        for b in "${B_ARR[@]}"; do
            i=$((i + 1))
            echo "[$i/$TOTAL_RUNS] starting FT_PE run" | tee -a "$LOG"
            run_ftpe "$b" "$t"
            cooldown "$COOLDOWN" "after run $i/$TOTAL_RUNS"
        done
    done
fi

banner "sweep complete"
echo "[done] $i runs.  log: $LOG" | tee -a "$LOG"

# Aggregate now so summary.txt is up to date.
if [ "$DRY_RUN" != "1" ]; then
    ./scripts/aggregate_results.sh 2>&1 | tee -a "$LOG" || true
fi
