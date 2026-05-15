"""PE-Core-L14-336 dynamic engine, 3 × batch=4 schedule, 100 iters.

Re-runs the "12 channels via 3 chained batch=4 inferences" scenario on the
**dynamic** TRT engine (min=1, opt=16, max=32), with:

  - 20 warmup iters (untimed)
  - 100 measured iters
  - per-iter timings persisted to a .log (CSV)
  - summary JSON includes mean, **std**, min, p50, p95, p99, max
    for both inference-only and full-cycle (preprocess + 3 infer + postprocess)

Run inside the Product-AI-mono conda env:
    python -m src.check_dynamic_3x4
"""

import json
import os
import sys
import time
from statistics import mean, stdev
from typing import List

import pandas as pd
import torch

from src.check_perception_encoder import (
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
from src.preprocess import load_image, preprocess

N_INFER = 3
BATCH = 4
WARMUP_ITERS = 20
MEASURE_ITERS = 100

RESULTS_JSON = "results/pe_check_dynamic_3x4.json"
RESULTS_LOG = "results/pe_check_dynamic_3x4.log"


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
    print(f"schedule: {N_INFER} x batch={BATCH} = {N_INFER * BATCH} channels/cycle")
    print(f"warmup={WARMUP_ITERS}  measure={MEASURE_ITERS}")

    ensure_onnx()
    export_trt_engine()

    model = TRTVisionEncoder(ENGINE_PATH)
    text_bank = make_text_bank()
    frame = load_image(SAMPLE_IMAGE)
    total_frames = N_INFER * BATCH

    # Preprocess once for the inference-only timing reuse.
    inp_full = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
    slices = [inp_full[i * BATCH : (i + 1) * BATCH].contiguous() for i in range(N_INFER)]

    for _ in range(WARMUP_ITERS):
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()

    inf_only_ms: List[float] = []
    full_ms: List[float] = []

    print("\nrunning ...")
    for it in range(MEASURE_ITERS):
        # 1) inference-only: GPU compute for the 3 chained batch=4 calls.
        t0 = time.perf_counter()
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf = (t1 - t0) * 1000

        # 2) full cycle: fresh preprocess + 3x infer + 3x postprocess.
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
        full = (t1 - t0) * 1000

        inf_only_ms.append(inf)
        full_ms.append(full)

        if (it + 1) % 10 == 0 or it == 0:
            print(
                f"  iter {it + 1:>3}/{MEASURE_ITERS}  "
                f"inference_only={inf:>7.2f} ms  full_cycle={full:>7.2f} ms"
            )

    os.makedirs(os.path.dirname(RESULTS_LOG) or ".", exist_ok=True)
    with open(RESULTS_LOG, "w") as f:
        f.write("iter,inference_only_ms,full_cycle_ms\n")
        for i, (a, b) in enumerate(zip(inf_only_ms, full_ms), start=1):
            f.write(f"{i},{a:.4f},{b:.4f}\n")
    print(f"\nWrote per-iter log: {RESULTS_LOG}")

    inf_stats = _stats(inf_only_ms)
    full_stats = _stats(full_ms)

    df = pd.DataFrame(
        [
            {"metric": "inference_only_ms", **inf_stats},
            {"metric": "full_cycle_ms", **full_stats},
        ]
    )
    print("\n" + df.to_string(index=False))

    payload = {
        **info,
        "engine_kind": "dynamic (min=1, opt=16, max=32)",
        "engine_path": ENGINE_PATH,
        "schedule": f"{N_INFER}x batch={BATCH}",
        "channels_per_cycle": total_frames,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "sample_image": SAMPLE_IMAGE,
        "log_file": RESULTS_LOG,
        "inference_only_ms": inf_stats,
        "full_cycle_ms": full_stats,
        "inference_only_meets_500ms": inf_stats["mean"] <= REALTIME_BUDGET_MS,
        "full_cycle_meets_500ms": full_stats["mean"] <= REALTIME_BUDGET_MS,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote summary JSON: {RESULTS_JSON}")

    inf_ok = payload["inference_only_meets_500ms"]
    full_ok = payload["full_cycle_meets_500ms"]
    print(
        f"\ninference_only ≤ 500 ms: {'PASS' if inf_ok else 'FAIL'} "
        f"(mean {inf_stats['mean']:.2f} ± {inf_stats['std']:.2f} ms)"
    )
    print(
        f"full_cycle     ≤ 500 ms: {'PASS' if full_ok else 'FAIL'} "
        f"(mean {full_stats['mean']:.2f} ± {full_stats['std']:.2f} ms)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
