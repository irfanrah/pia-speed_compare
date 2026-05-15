"""PE-Core-L14-336 chained-batch checker.

Tests the claim from the report: "A4000 handles 12 channels in 0.5 s real time"
via several scheduling policies, all of which process 12 frames per cycle:

    3 inferences of batch=4   (claimed scheme)
    4 inferences of batch=3
    6 inferences of batch=2
    12 inferences of batch=1  (the no-batch baseline; thermal-prone)

For each schedule we time both:
    inference_only_ms  GPU compute only (preprocess done once, before timing)
    full_cycle_ms      preprocess(12) + N chained inferences + postprocess(N)

This isolates "the report's number" (inference-only) from the wall-clock a
realtime system sees.
"""

import json
import os
import sys
import time
from typing import List, Tuple

import pandas as pd
import torch

from src.check_perception_encoder import (
    ENGINE_PATH,
    IMG_SIZE,
    INPUT_SIZE,
    REALTIME_BUDGET_MS,
    SAMPLE_IMAGE,
    TARGET_CHANNELS,
    TRT_PROFILE,
    TRTVisionEncoder,
    _percentile,
    device_info,
    ensure_onnx,
    export_trt_engine,
    make_text_bank,
    postprocess,
)
from src.preprocess import load_image, preprocess

SCHEDULES: List[Tuple[int, int]] = [
    (3, 4),
    (4, 3),
    (6, 2),
    (12, 1),
]

WARMUP_ITERS = 10
MEASURE_ITERS = 30
RESULTS_JSON = "results/pe_check_chained.json"


@torch.inference_mode()
def measure_schedule(
    model: TRTVisionEncoder,
    text_bank: torch.Tensor,
    frame,
    n_infer: int,
    batch: int,
) -> dict:
    total_frames = n_infer * batch
    inp_full = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
    slices = [inp_full[i * batch : (i + 1) * batch].contiguous() for i in range(n_infer)]

    for _ in range(WARMUP_ITERS):
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()

    inf_only_ms = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_only_ms.append((t1 - t0) * 1000)

    full_ms = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        inp = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
        slices_now = [inp[i * batch : (i + 1) * batch].contiguous() for i in range(n_infer)]
        for s in slices_now:
            feats = model(s)
            _ = postprocess(feats, text_bank)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        full_ms.append((t1 - t0) * 1000)

    return {
        "schedule": f"{n_infer}x batch={batch}",
        "n_infer": n_infer,
        "batch": batch,
        "channels_per_cycle": total_frames,
        "inference_only_ms_mean": round(sum(inf_only_ms) / len(inf_only_ms), 3),
        "inference_only_ms_p95": round(_percentile(inf_only_ms, 95), 3),
        "full_cycle_ms_mean": round(sum(full_ms) / len(full_ms), 3),
        "full_cycle_ms_p95": round(_percentile(full_ms, 95), 3),
        "inference_only_meets_500ms": (sum(inf_only_ms) / len(inf_only_ms)) <= REALTIME_BUDGET_MS,
        "full_cycle_meets_500ms": (sum(full_ms) / len(full_ms)) <= REALTIME_BUDGET_MS,
    }


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

    ensure_onnx()
    export_trt_engine()

    model = TRTVisionEncoder(ENGINE_PATH)
    text_bank = make_text_bank()
    frame = load_image(SAMPLE_IMAGE)

    rows = []
    for n_infer, batch in SCHEDULES:
        print(
            f"================================== Start Schedule: {n_infer}x batch={batch} =================================="
        )
        r = measure_schedule(model, text_bank, frame, n_infer, batch)
        rows.append(r)
        df = pd.DataFrame([r])[
            [
                "schedule",
                "channels_per_cycle",
                "inference_only_ms_mean",
                "inference_only_ms_p95",
                "full_cycle_ms_mean",
                "full_cycle_ms_p95",
                "inference_only_meets_500ms",
                "full_cycle_meets_500ms",
            ]
        ]
        print(df.to_string(index=False))
        print(
            f"================================== End Schedule: {n_infer}x batch={batch} =================================="
        )

    payload = {
        **info,
        "precision": "fp16",
        "input_shape": list(INPUT_SIZE),
        "engine_profile": TRT_PROFILE,
        "target_channels": TARGET_CHANNELS,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "sample_image": SAMPLE_IMAGE,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "results": rows,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")

    inf_pass = [r for r in rows if r["inference_only_meets_500ms"]]
    full_pass = [r for r in rows if r["full_cycle_meets_500ms"]]
    print(f"\ninference-only ≤ 500 ms: {[r['schedule'] for r in inf_pass]}")
    print(f"full cycle    ≤ 500 ms: {[r['schedule'] for r in full_pass]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
