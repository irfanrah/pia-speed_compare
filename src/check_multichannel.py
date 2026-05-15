"""PE-Core-L14-336 multi-channel checker (batch=1 sequential).

Companion to ``check_perception_encoder.py``. Instead of stuffing N channels
into one batch=N inference, this checker runs **N back-to-back batch=1
inferences per cycle** — modelling the case where each CCTV channel is
processed one image at a time. We report:

    cycle_total_ms     wall time for one full cycle of N channels
    per_channel_ms     cycle_total_ms / N  (effective per-frame latency)
    fps_aggregate      N * 1000 / cycle_total_ms

If ``cycle_total_ms`` <= 500, the GPU can refresh every channel within the
realtime budget at this scheduling policy.

Runs inside the Product-AI-mono conda env.
"""

import json
import os
import sys
import time
from typing import List

import pandas as pd
import torch

from src.check_perception_encoder import (
    ENGINE_PATH,
    IMG_SIZE,
    INPUT_SIZE,
    MEASURE_ITERS,
    REALTIME_BUDGET_MS,
    SAMPLE_IMAGE,
    TARGET_CHANNELS,
    TRT_PROFILE,
    TRTVisionEncoder,
    WARMUP_ITERS,
    _percentile,
    device_info,
    ensure_onnx,
    export_trt_engine,
    make_text_bank,
    postprocess,
)
from src.preprocess import load_image, preprocess

CHANNEL_COUNTS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24]
RESULTS_JSON = "results/pe_check_multichannel.json"


@torch.inference_mode()
def test_channels_batch1(
    image_path: str = SAMPLE_IMAGE,
    channel_counts: List[int] = CHANNEL_COUNTS,
) -> List[dict]:
    model = TRTVisionEncoder(ENGINE_PATH)
    text_bank = make_text_bank()
    frame = load_image(image_path)

    rows: List[dict] = []
    for n in channel_counts:
        print(
            f"================================== Start Test Channels (batch=1 serial): {n} =================================="
        )

        for _ in range(WARMUP_ITERS):
            for _ in range(n):
                inp = preprocess([frame], resize_size=IMG_SIZE, device="cuda")
                feats = model(inp)
                _ = postprocess(feats, text_bank)
        torch.cuda.synchronize()

        cycle_ms = []
        for _ in range(MEASURE_ITERS):
            t0 = time.perf_counter()
            for _ in range(n):
                inp = preprocess([frame], resize_size=IMG_SIZE, device="cuda")
                feats = model(inp)
                _ = postprocess(feats, text_bank)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            cycle_ms.append((t1 - t0) * 1000)

        mean = sum(cycle_ms) / len(cycle_ms)
        p95 = _percentile(cycle_ms, 95)
        row = {
            "channels": n,
            "cycle_total_ms": round(mean, 3),
            "per_channel_ms": round(mean / n, 3),
            "p95_cycle_ms": round(p95, 3),
            "fps_aggregate": round(n * 1000.0 / mean, 2),
            "meets_500ms_cycle": mean <= REALTIME_BUDGET_MS,
        }
        df = pd.DataFrame([row])
        print(df.to_string(index=False))
        print(
            f"================================== End Test Channels (batch=1 serial): {n} =================================="
        )
        rows.append(row)

    return rows


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

    rows = test_channels_batch1(SAMPLE_IMAGE, CHANNEL_COUNTS)
    target = next((r for r in rows if r["channels"] == TARGET_CHANNELS), None)
    target_pass = bool(target and target["meets_500ms_cycle"])

    payload = {
        **info,
        "precision": "fp16",
        "input_shape": list(INPUT_SIZE),
        "engine_profile": TRT_PROFILE,
        "scheduling": "batch=1 sequential per cycle",
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "target_channels": TARGET_CHANNELS,
        "sample_image": SAMPLE_IMAGE,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "results": rows,
        "target_result": target,
        "pass": target_pass,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")

    msg = "PASS" if target_pass else "FAIL"
    if target:
        print(
            f"{msg}: {TARGET_CHANNELS} channels cycle={target['cycle_total_ms']:.2f} ms "
            f"(per-channel {target['per_channel_ms']:.2f} ms, budget {REALTIME_BUDGET_MS:.0f} ms)"
        )
    return 0 if target_pass else 1


if __name__ == "__main__":
    sys.exit(main())
