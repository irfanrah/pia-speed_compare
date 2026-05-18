"""PE-Core-L14-336 dynamic engine, 3 × batch=4, short cold-state run.

5 warmup iters, 20 measured iters. Captures both inference-only and full-cycle
timings plus GPU temp. Plot shows iter on Y, inference + full_cycle on bottom X,
GPU temperature on top X.

Run inside the Product-AI-mono conda env:
    python -m src.check_dynamic_3x4_short
"""

import json
import os
import sys
import time
from statistics import mean, stdev
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.check_dynamic_3x4_thermal import gpu_temp_c  # noqa: E402
from src.check_perception_encoder import (  # noqa: E402
    ENGINE_PATH,
    IMG_SIZE,
    INPUT_SIZE,
    REALTIME_BUDGET_MS,
    SAMPLE_IMAGE,
    TARGET_CHANNELS,
    TRTVisionEncoder,
    _percentile,
    device_info,
    ensure_onnx,
    export_trt_engine,
    make_text_bank,
    postprocess,
)
from src.preprocess import load_image, preprocess  # noqa: E402

N_INFER = 3
BATCH = 4
WARMUP_ITERS = 5
MEASURE_ITERS = 25

OUT_DIR = "results"
LOG_ITER = f"{OUT_DIR}/pe_check_dynamic_3x4_short.log"
JSON_OUT = f"{OUT_DIR}/pe_check_dynamic_3x4_short.json"
PNG_OUT = f"{OUT_DIR}/pe_check_dynamic_3x4_short.png"


def _stats(values: List[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(mean(values), 3),
        "std": round(stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


@torch.inference_mode()
def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    info = device_info()
    print(f"device: {info['device']}")
    print(
        f"  index={info['device_index']}/{info['device_count']}  "
        f"sm={info['compute_capability']}  mem={info['total_memory_mib']} MiB"
    )
    print(
        f"  torch={info['torch_version']}  tensorrt={info['tensorrt_version']}  "
        f"driver={info['driver_version']}  cuda={info['cuda_runtime']}"
    )
    print(f"engine: dynamic ({ENGINE_PATH})")
    print(
        f"schedule: {N_INFER} x batch={BATCH} = {N_INFER * BATCH} channels/cycle, "
        f"warmup={WARMUP_ITERS} measure={MEASURE_ITERS}"
    )

    ensure_onnx()
    export_trt_engine()

    model = TRTVisionEncoder(ENGINE_PATH)
    text_bank = make_text_bank()
    frame = load_image(SAMPLE_IMAGE)
    total_frames = N_INFER * BATCH

    inp_full = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
    slices = [
        inp_full[i * BATCH : (i + 1) * BATCH].contiguous() for i in range(N_INFER)
    ]

    print(f"starting temp: {gpu_temp_c()}°C")
    for _ in range(WARMUP_ITERS):
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()
    print(f"post-warmup temp: {gpu_temp_c()}°C")

    iter_rows: List[dict] = []
    t_start = time.perf_counter()

    for it in range(1, MEASURE_ITERS + 1):
        t0 = time.perf_counter()
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_only_ms = (t1 - t0) * 1000

        t0 = time.perf_counter()
        inp = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
        slices_now = [
            inp[i * BATCH : (i + 1) * BATCH].contiguous() for i in range(N_INFER)
        ]
        for s in slices_now:
            feats = model(s)
            _ = postprocess(feats, text_bank)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        full_ms = (t1 - t0) * 1000

        temp = gpu_temp_c()
        iter_rows.append(
            {
                "iter": it,
                "elapsed_s": round(time.perf_counter() - t_start, 3),
                "inference_only_ms": round(inf_only_ms, 4),
                "full_cycle_ms": round(full_ms, 4),
                "gpu_temp_c": temp,
            }
        )
        print(
            f"  iter {it:>2}/{MEASURE_ITERS}  "
            f"inf={inf_only_ms:>7.2f} ms  full={full_ms:>7.2f} ms  temp={temp}°C"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_ITER, "w") as f:
        f.write("iter,elapsed_s,inference_only_ms,full_cycle_ms,gpu_temp_c\n")
        for r in iter_rows:
            f.write(
                f"{r['iter']},{r['elapsed_s']},{r['inference_only_ms']:.4f},"
                f"{r['full_cycle_ms']:.4f},{r['gpu_temp_c']}\n"
            )
    print(f"\nwrote {LOG_ITER}")

    inf_vals = [r["inference_only_ms"] for r in iter_rows]
    full_vals = [r["full_cycle_ms"] for r in iter_rows]
    temp_vals = [r["gpu_temp_c"] for r in iter_rows]
    inf_stats = _stats(inf_vals)
    full_stats = _stats(full_vals)

    summary = {
        **info,
        "engine_kind": "dynamic (min=1, opt=16, max=32)",
        "engine_path": ENGINE_PATH,
        "schedule": f"{N_INFER}x batch={BATCH}",
        "channels_per_cycle": total_frames,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "inference_only_ms": inf_stats,
        "full_cycle_ms": full_stats,
        "gpu_temp_c_min": min(temp_vals),
        "gpu_temp_c_max": max(temp_vals),
        "log_files": {"iter": LOG_ITER, "plot": PNG_OUT},
        "inference_only_meets_500ms": inf_stats["mean"] <= REALTIME_BUDGET_MS,
        "full_cycle_meets_500ms": full_stats["mean"] <= REALTIME_BUDGET_MS,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {JSON_OUT}")

    iters = [r["iter"] for r in iter_rows]
    fig, ax_full = plt.subplots(figsize=(10, 6))

    ax_full.plot(iters, full_vals, "s-", color="tab:orange", markersize=5, label="full_cycle_ms")
    ax_full.axhline(REALTIME_BUDGET_MS, color="gray", linestyle=":", alpha=0.6, label="500 ms budget")
    ax_full.set_xlabel("iteration")
    ax_full.set_ylabel("full_cycle_ms", color="tab:orange")
    ax_full.tick_params(axis="y", labelcolor="tab:orange")
    ax_full.grid(True, alpha=0.25)

    ax_t = ax_full.twinx()
    ax_t.plot(iters, temp_vals, "^-", color="tab:blue", markersize=4, alpha=0.8, label="gpu_temp_C")
    ax_t.set_ylabel("GPU temperature (°C)", color="tab:blue")
    ax_t.tick_params(axis="y", labelcolor="tab:blue")

    h1, l1 = ax_full.get_legend_handles_labels()
    h2, l2 = ax_t.get_legend_handles_labels()
    ax_full.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    plt.title(
        f"PE-Core-L14-336 dynamic  {N_INFER}×batch={BATCH}  "
        f"({WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iters)\n"
        f"full_cycle mean={full_stats['mean']:.1f} ± {full_stats['std']:.1f} ms  "
        f"(min={full_stats['min']:.1f}  p95={full_stats['p95']:.1f})"
    )
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=120, bbox_inches="tight")
    print(f"wrote {PNG_OUT}")

    print(
        f"\ninference_only:  mean {inf_stats['mean']:.2f} ± {inf_stats['std']:.2f} ms  "
        f"(min {inf_stats['min']:.2f}, p95 {inf_stats['p95']:.2f})  "
        f"{'PASS' if summary['inference_only_meets_500ms'] else 'FAIL'}"
    )
    print(
        f"full_cycle:      mean {full_stats['mean']:.2f} ± {full_stats['std']:.2f} ms  "
        f"(min {full_stats['min']:.2f}, p95 {full_stats['p95']:.2f})  "
        f"{'PASS' if summary['full_cycle_meets_500ms'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
