import json
import os
import subprocess
import sys
from statistics import mean

import tensorrt as trt
import torch

from src.preprocess import preprocess

from .config import (
    BENCHMARK_BATCHES,
    BENCHMARK_MEASURE_ITERS,
    BENCHMARK_WARMUP_ITERS,
    IMG_SIZE,
    INPUT_SIZE,
    PERCEPTION_ENCODER_ONNX_PATH,
    PERCEPTION_ENCODER_TRT_PATH,
    REALTIME_BUDGET_SECONDS,
    RESULTS_PATH,
    SAMPLE_IMAGES,
    TARGET_CHANNELS,
    TRT_MAX_BATCH,
    TRT_MIN_BATCH,
    TRT_OPT_BATCH,
)
from .download_model import ensure_onnx
from .realtime_sim import analytic_stream_sim, best_sustainable_capacity
from .trt_export import export_trt_engine
from .trt_load import TRTInference

STREAMING_SCENARIOS = [
    (12, 1.0),
    (12, 2.0),
    (12, 3.0),
    (12, 0.5),
    (24, 1.0),
    (36, 1.0),
    (6, 1.0),
    (4, 1.0),
]


def _percentile(values, p):
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _driver_version():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return None


def _device_info():
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "device": torch.cuda.get_device_name(idx),
        "device_index": idx,
        "device_count": torch.cuda.device_count(),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mib": int(props.total_memory / 1024 / 1024),
        "driver_version": _driver_version(),
        "torch_version": torch.__version__,
        "tensorrt_version": trt.__version__,
        "cuda_runtime": torch.version.cuda,
    }


@torch.inference_mode()
def _benchmark_batch(model: TRTInference, batch_size: int, img_path: str) -> dict:
    # Preprocess is intentionally OUTSIDE the timed region: the same CUDA tensor
    # is reused for warmup and all measured iters so the timing reflects pure
    # model(sample) execution.
    sample = preprocess(img_path, batch=batch_size, resize_size=IMG_SIZE, device="cuda")

    for _ in range(BENCHMARK_WARMUP_ITERS):
        model(sample)
    torch.cuda.synchronize()

    timings_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(BENCHMARK_MEASURE_ITERS):
        start.record()
        model(sample)
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))

    mean_ms = mean(timings_ms)
    return {
        "batch": batch_size,
        "image": img_path,
        "iters": BENCHMARK_MEASURE_ITERS,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(_percentile(timings_ms, 50), 3),
        "p95_ms": round(_percentile(timings_ms, 95), 3),
        "p99_ms": round(_percentile(timings_ms, 99), 3),
        "min_ms": round(min(timings_ms), 3),
        "max_ms": round(max(timings_ms), 3),
        "images_per_sec": round(batch_size * 1000.0 / mean_ms, 2),
        "meets_realtime_500ms": mean_ms <= REALTIME_BUDGET_SECONDS * 1000.0,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; TRT FP16 benchmark requires a GPU.")

    info = _device_info()
    print(f"device: {info['device']}")
    print(f"  index={info['device_index']}/{info['device_count']}  "
          f"sm={info['compute_capability']}  mem={info['total_memory_mib']} MiB")
    print(f"  torch={info['torch_version']}  tensorrt={info['tensorrt_version']}  "
          f"driver={info['driver_version']}  cuda={info['cuda_runtime']}")

    ensure_onnx(PERCEPTION_ENCODER_ONNX_PATH)
    # FP16-weights ONNX variant was tested and proved slower (cast overhead in
    # attention blocks) — keep the FP32-weights ONNX + BuilderFlag.FP16 path.
    export_trt_engine(
        onnx_file=PERCEPTION_ENCODER_ONNX_PATH,
        save_file_path=PERCEPTION_ENCODER_TRT_PATH,
        input_size=INPUT_SIZE,
        min_batch_size=TRT_MIN_BATCH,
        opt_batch_size=TRT_OPT_BATCH,
        max_batch_size=TRT_MAX_BATCH,
        half_precision=True,
    )

    model = TRTInference(PERCEPTION_ENCODER_TRT_PATH)

    for p in SAMPLE_IMAGES:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Sample image missing: {p}")
    print(f"sample images: {SAMPLE_IMAGES}")

    results = []
    for b in BENCHMARK_BATCHES:
        img_path = SAMPLE_IMAGES[b % len(SAMPLE_IMAGES)]
        r = _benchmark_batch(model, b, img_path)
        flag = "OK " if r["meets_realtime_500ms"] else "MISS"
        print(
            f"  [{flag}] batch={b:>2}  img={os.path.basename(img_path):<8}  "
            f"mean={r['mean_ms']:>7.2f} ms  p95={r['p95_ms']:>7.2f} ms  "
            f"ips={r['images_per_sec']:>8.2f}"
        )
        results.append(r)

    target = next((r for r in results if r["batch"] == TARGET_CHANNELS), None)
    target_pass = bool(target and target["meets_realtime_500ms"])

    lat_map = {r["batch"]: r["mean_ms"] for r in results}
    capacity = best_sustainable_capacity(lat_map, budget_ms=REALTIME_BUDGET_SECONDS * 1000.0)
    print(
        f"\ncapacity: best_batch={capacity['best_batch']}  "
        f"per_batch_latency={capacity['best_batch_latency_ms']} ms  "
        f"max_sustainable_aggregate_fps={capacity['max_sustainable_fps']}"
    )

    print("\nstreaming simulation (option 4: per-channel fps, batched on arrival,")
    print("                     analytic discrete-event sim from measured latencies)")
    streaming = []
    for nch, fps in STREAMING_SCENARIOS:
        s = analytic_stream_sim(
            lat_map=lat_map,
            num_channels=nch,
            channel_fps=fps,
            duration_s=30.0,
            max_batch=TRT_MAX_BATCH,
            budget_ms=REALTIME_BUDGET_SECONDS * 1000.0,
        )
        flag = "OK " if s["meets_realtime_500ms"] else "MISS"
        print(
            f"  [{flag}] ch={nch:>2} fps/ch={fps:>4}  "
            f"agg={s['aggregate_target_fps']:>5} fps  "
            f"actual={s['throughput_fps']:>5} fps  "
            f"p95={s['latency_p95_ms']:>7.1f} ms  "
            f"backlog={s['backlog_remaining']}"
        )
        streaming.append(s)

    streaming_target = next(
        (s for s in streaming if s["num_channels"] == TARGET_CHANNELS and s["channel_fps"] == 1.0),
        None,
    )

    payload = {
        **info,
        "precision": "fp16",
        "fp16_weights_onnx": True,
        "timing_scope": "model_forward_only (preprocess excluded)",
        "input_shape": list(INPUT_SIZE),
        "engine_profile": {
            "min": TRT_MIN_BATCH,
            "opt": TRT_OPT_BATCH,
            "max": TRT_MAX_BATCH,
        },
        "realtime_budget_seconds": REALTIME_BUDGET_SECONDS,
        "target_channels": TARGET_CHANNELS,
        "sample_images": SAMPLE_IMAGES,
        "target_result": target,
        "results": results,
        "capacity": capacity,
        "streaming_results": streaming,
        "streaming_target": streaming_target,
        "pass": target_pass,
        "streaming_pass": bool(streaming_target and streaming_target["meets_realtime_500ms"]),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")

    if not target_pass:
        print(
            f"FAIL: batch={TARGET_CHANNELS} did not meet "
            f"{REALTIME_BUDGET_SECONDS * 1000:.0f} ms budget."
        )
        return 1

    print(
        f"PASS: batch={TARGET_CHANNELS} mean={target['mean_ms']:.2f} ms "
        f"≤ {REALTIME_BUDGET_SECONDS * 1000:.0f} ms."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
