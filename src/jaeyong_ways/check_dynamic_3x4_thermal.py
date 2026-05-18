"""PE-Core-L14-336 dynamic engine, 3 × batch=4 schedule, 100 iters with mid-run rest.

Like check_dynamic_3x4.py, but also:
  - reads GPU temperature after every iteration
  - rests 60 s after iter 50 (with temp sampled every 2 s)
  - saves a plot: iter (Y, inverted) vs inference_only_ms (bottom X)
                  and GPU temp (top X)

Outputs:
  results/pe_check_dynamic_3x4_thermal.log        per-iter CSV
  results/pe_check_dynamic_3x4_thermal_rest.log   rest-period temp CSV
  results/pe_check_dynamic_3x4_thermal.json       summary
  results/pe_check_dynamic_3x4_thermal.png        plot
"""

import json
import os
import subprocess
import sys
import time
from statistics import mean, stdev
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

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
WARMUP_ITERS = 20
MEASURE_ITERS = 100
REST_AFTER_ITER = 50
REST_SECONDS = 60
REST_POLL_INTERVAL_S = 2.0
GPU_INDEX = 0

OUT_DIR = "results"
LOG_ITER = f"{OUT_DIR}/pe_check_dynamic_3x4_thermal.log"
LOG_REST = f"{OUT_DIR}/pe_check_dynamic_3x4_thermal_rest.log"
JSON_OUT = f"{OUT_DIR}/pe_check_dynamic_3x4_thermal.json"
PNG_OUT = f"{OUT_DIR}/pe_check_dynamic_3x4_thermal.png"


def gpu_temp_c(idx: int = GPU_INDEX) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={idx}",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(out.splitlines()[0])


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
        f"{MEASURE_ITERS} iters, rest {REST_SECONDS}s after iter {REST_AFTER_ITER}"
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

    for _ in range(WARMUP_ITERS):
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()

    iter_rows: List[dict] = []
    rest_rows: List[dict] = []

    print(f"starting temp: {gpu_temp_c()}°C")
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
        elapsed = time.perf_counter() - t_start
        iter_rows.append(
            {
                "iter": it,
                "elapsed_s": round(elapsed, 3),
                "inference_only_ms": round(inf_only_ms, 4),
                "full_cycle_ms": round(full_ms, 4),
                "gpu_temp_c": temp,
            }
        )

        if it % 10 == 0 or it == 1 or it == REST_AFTER_ITER:
            print(
                f"  iter {it:>3}/{MEASURE_ITERS}  "
                f"inf={inf_only_ms:>7.2f} ms  full={full_ms:>7.2f} ms  temp={temp}°C"
            )

        if it == REST_AFTER_ITER:
            print(f"\n--- resting {REST_SECONDS}s after iter {it} ---")
            rest_t0 = time.perf_counter()
            while True:
                elapsed_rest = time.perf_counter() - rest_t0
                if elapsed_rest >= REST_SECONDS:
                    break
                t_now = time.perf_counter()
                rest_rows.append(
                    {
                        "elapsed_s_in_rest": round(elapsed_rest, 2),
                        "elapsed_s_total": round(t_now - t_start, 2),
                        "gpu_temp_c": gpu_temp_c(),
                    }
                )
                if int(elapsed_rest) % 10 == 0:
                    print(
                        f"    rest +{elapsed_rest:>5.1f}s  temp={rest_rows[-1]['gpu_temp_c']}°C"
                    )
                time.sleep(REST_POLL_INTERVAL_S)
            final_temp = gpu_temp_c()
            print(f"--- rest done, temp now {final_temp}°C ---\n")
            rest_rows.append(
                {
                    "elapsed_s_in_rest": REST_SECONDS,
                    "elapsed_s_total": round(time.perf_counter() - t_start, 2),
                    "gpu_temp_c": final_temp,
                }
            )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_ITER, "w") as f:
        f.write("iter,elapsed_s,inference_only_ms,full_cycle_ms,gpu_temp_c\n")
        for r in iter_rows:
            f.write(
                f"{r['iter']},{r['elapsed_s']},{r['inference_only_ms']:.4f},"
                f"{r['full_cycle_ms']:.4f},{r['gpu_temp_c']}\n"
            )
    print(f"wrote {LOG_ITER}")

    with open(LOG_REST, "w") as f:
        f.write("elapsed_s_in_rest,elapsed_s_total,gpu_temp_c\n")
        for r in rest_rows:
            f.write(
                f"{r['elapsed_s_in_rest']},{r['elapsed_s_total']},{r['gpu_temp_c']}\n"
            )
    print(f"wrote {LOG_REST}")

    inf_vals = [r["inference_only_ms"] for r in iter_rows]
    full_vals = [r["full_cycle_ms"] for r in iter_rows]
    temp_vals = [r["gpu_temp_c"] for r in iter_rows]
    inf_pre = [r["inference_only_ms"] for r in iter_rows if r["iter"] <= REST_AFTER_ITER]
    inf_post = [r["inference_only_ms"] for r in iter_rows if r["iter"] > REST_AFTER_ITER]

    summary = {
        **info,
        "engine_kind": "dynamic (min=1, opt=16, max=32)",
        "engine_path": ENGINE_PATH,
        "schedule": f"{N_INFER}x batch={BATCH}",
        "channels_per_cycle": total_frames,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "rest_after_iter": REST_AFTER_ITER,
        "rest_seconds": REST_SECONDS,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "log_files": {"iter": LOG_ITER, "rest": LOG_REST, "plot": PNG_OUT},
        "inference_only_ms": _stats(inf_vals),
        "full_cycle_ms": _stats(full_vals),
        "inference_only_ms_pre_rest": _stats(inf_pre),
        "inference_only_ms_post_rest": _stats(inf_post),
        "gpu_temp_c_min": min(temp_vals),
        "gpu_temp_c_max": max(temp_vals),
    }
    with open(JSON_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {JSON_OUT}")

    iters = [r["iter"] for r in iter_rows]
    fig, ax_ms = plt.subplots(figsize=(9, 11))
    ax_ms.plot(inf_vals, iters, "o-", color="tab:red", markersize=3, label="inference_only_ms")
    ax_ms.set_xlabel("inference_only_ms", color="tab:red")
    ax_ms.set_ylabel("iteration")
    ax_ms.tick_params(axis="x", labelcolor="tab:red")
    ax_ms.invert_yaxis()
    ax_ms.axvline(REALTIME_BUDGET_MS, color="gray", linestyle=":", alpha=0.6, label="500 ms budget")
    ax_ms.axhline(
        REST_AFTER_ITER + 0.5,
        color="tab:green",
        linestyle="--",
        alpha=0.7,
        label=f"{REST_SECONDS}s rest after iter {REST_AFTER_ITER}",
    )

    ax_t = ax_ms.twiny()
    ax_t.plot(temp_vals, iters, "s-", color="tab:blue", markersize=3, label="gpu_temp_C")
    ax_t.set_xlabel("GPU temperature (°C)", color="tab:blue")
    ax_t.tick_params(axis="x", labelcolor="tab:blue")

    h1, l1 = ax_ms.get_legend_handles_labels()
    h2, l2 = ax_t.get_legend_handles_labels()
    ax_ms.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

    plt.title(
        f"PE-Core-L14-336 dynamic {N_INFER}×batch={BATCH}  ({MEASURE_ITERS} iters, "
        f"{REST_SECONDS}s rest after iter {REST_AFTER_ITER})"
    )
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=120)
    print(f"wrote {PNG_OUT}")

    s_pre = _stats(inf_pre) if inf_pre else None
    s_post = _stats(inf_post) if inf_post else None
    print(
        f"\npre-rest  (iters 1..{REST_AFTER_ITER}):  "
        f"mean {s_pre['mean']:.2f} ± {s_pre['std']:.2f} ms"
    )
    print(
        f"post-rest (iters {REST_AFTER_ITER + 1}..{MEASURE_ITERS}): "
        f"mean {s_post['mean']:.2f} ± {s_post['std']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
