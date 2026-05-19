"""Speed benchmark for Qwen3-VL-Embedding (async / remote vLLM-server variant).

Instantiates ``pia_prod.AI.modules.qwen3vle.service.Qwen3VLEService`` -- the
remote variant where the heavy embedding step is a JPEG-encoded HTTP POST to
a separate vLLM ``embedding_server`` container.

Inputs are **synthetic random ndarrays** generated in-memory at startup. No
JPEG decode from disk, no real image file involved at any point. Every
iteration sees different pixel values (rotated round-robin through a pool),
which is necessary to bypass the remote vLLM server's `mm_processor_cache`
and measure the honest per-call model cost.

Timing strategy
---------------
Each iteration runs the pipeline ONCE end-to-end with a `time_call` wrapped
around each of 4 ordered steps. Input prep (memcpy from the random pool) is
deliberately untimed -- the measurement reflects only the GPU/CPU compute
pipeline plus the HTTP roundtrip.

Stage totals are derived by summing the relevant step times per iteration:

    full_cycle = preprocess + model + postprocess + alarm_gate
    half_cycle = preprocess + model + postprocess        (= full - alarm_gate)
    inference  = model

This guarantees `full >= half >= inference` per iteration -- no variance
artefacts where a stage came out larger than its superset.

Step boundaries
---------------
    preprocess  : cv_bgr2rgb -> roi_crop -> resize -> buffer -> extract_ready
    model       : _embed_via_server
                   (= JPEG encode + base64 + HTTP POST /v1/embeddings/batch
                    + parse response). This is the call that lives on the
                    remote container; local-side cost is encode + HTTP only.
    postprocess : _get_category_predictions
                   (F.normalize + cos-sim matmul + verdict)
    alarm_gate  : alarm_event_manager.update (duration queue + alarm gating)

GPU temperature here reflects the LOCAL GPU's load (preprocess + cos-sim
only). The remote container's GPU is not visible from this process.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "Product-AI-mono" / "packages"))

DEFAULT_TXT = REPO_ROOT / "assets" / "model" / "Qwen3-VL-Embedding-2B-FP8" / "VLE_FP8_text_features.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_SERVER_URL = "http://localhost:8210"

# Synthetic input resolution. Defaults match a typical 1080p camera frame so
# preprocess (roi + resize to 768^2) does comparable work to a realistic
# production source.
DEFAULT_INPUT_H = 1080
DEFAULT_INPUT_W = 1920
DEFAULT_INPUT_C = 3


# ---- Env-var injection BEFORE importing Qwen3VLEService ---------------------
# pia_prod.AI.modules.qwen3vle.config reads these at import time, so we have to
# set them before the import chain starts.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL)
_pre.add_argument("--text-features", type=Path, default=DEFAULT_TXT)
_pre.add_argument("--timeout", type=float, default=30.0)
_pre.add_argument("--jpeg-quality", type=int, default=95)
_pre_args, _ = _pre.parse_known_args()
os.environ["QWEN3VLE_EMBEDDING_SERVER_URL"] = _pre_args.server_url
os.environ["QWEN3VLE_TEXT_FEATURES_PATH"] = str(_pre_args.text_features)
os.environ["QWEN3VLE_EMBEDDING_TIMEOUT"] = str(_pre_args.timeout)
os.environ["QWEN3VLE_EMBEDDING_JPEG_QUALITY"] = str(_pre_args.jpeg_quality)
# -----------------------------------------------------------------------------

from queue import Queue  # noqa: E402

from pia.ai.tasks.T2VRet.models.PE.utils.complexity_check import (  # noqa: E402
    time_call,
    get_gpu_stats_nvml,
)
from pia.vision.preprocessing import cv_bgr2rgb_batch  # noqa: E402
from pia_prod.AI.modules.qwen3vle.service import Qwen3VLEService  # noqa: E402
from pia_prod.AI.modules.qwen3vle.config import IMG_SIZE, TEMPORAL_SIZE, QWEN3VLE_ID  # noqa: E402


def query_gpu_temp_c(device_index: int = 0) -> float | None:
    stats = get_gpu_stats_nvml(f"cuda:{device_index}")
    if stats and "temp_C" in stats:
        return float(stats["temp_C"])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def gpu_info(device_index: int = 0) -> dict:
    props = torch.cuda.get_device_properties(device_index)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        driver = None
    return {
        "name": torch.cuda.get_device_name(device_index),
        "device_index": device_index,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mib": int(props.total_memory / 1024 / 1024),
        "driver_version": driver,
    }


def stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    def pct(p: float) -> float:
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        return s[k]
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "median_ms": round(statistics.median(samples_ms), 3),
        "p95_ms": round(pct(95), 3),
        "p99_ms": round(pct(99), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "std_ms": round(statistics.pstdev(samples_ms), 3) if len(samples_ms) > 1 else 0.0,
    }


def make_user_params(batch_size: int) -> list[dict]:
    """retEvent must be a dict keyed by category id because the ROI manager
    indexes into it; empty polygonCoordinates falls back to full frame."""
    return [
        {"user_param": {
            "retEvent": {
                "fire_vle_ret": {"roi": {"polygonCoordinates": []}},
            },
            "cameraId": f"cam_{i}",
            "organization": "pia",
        }}
        for i in range(batch_size)
    ]


@torch.inference_mode()
def benchmark(
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    input_h: int = DEFAULT_INPUT_H,
    input_w: int = DEFAULT_INPUT_W,
    input_c: int = DEFAULT_INPUT_C,
    random_pool_size: int = 64,
) -> dict:
    service = Qwen3VLEService(Queue())
    user_params = make_user_params(batch_size)
    stream_ids = [f"stream_{i}" for i in range(batch_size)]

    # Pre-generate a pool of random uint8 ndarrays at the requested H/W/C.
    # Every call to `next_input()` returns a fresh copy of a *different* pool
    # entry, so every iteration sees different pixel values -- this bypasses
    # the remote server's mm_processor_cache and forces the vision tower to
    # run for real on every call. Pool is rotated round-robin; pool_size >=
    # batch_size * 2 guarantees no within-iteration duplication.
    rng = np.random.default_rng(seed=0)
    random_pool = [
        rng.integers(0, 256, size=(input_h, input_w, input_c), dtype=np.uint8)
        for _ in range(max(random_pool_size, batch_size * 2))
    ]
    _idx = [0]

    def next_input() -> np.ndarray:
        arr = random_pool[_idx[0] % len(random_pool)]
        _idx[0] += 1
        return arr.copy()

    # Warmup — exercises the full _detect chain so the first measured iter
    # doesn't pay for remote-server JIT / CUDA graph capture or HTTP cold
    # connection costs.
    for _ in range(warmup_iters):
        batches = [next_input() for _ in range(batch_size)]
        if not service.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)
        cropped = service.roi_manager.process_batches_with_roi(batches, user_params)
        for sid, frame in zip(stream_ids, cropped):
            service.frame_buffers[sid].append(TF.resize(frame, service.img_size, antialias=True))
        ready = service._extract_ready_videos(stream_ids, batches, user_params)
        if ready is not None:
            batched_videos, ready_sids, ready_user_params, _ = ready
            vid_embeddings = service._embed_via_server(batched_videos)
            predictions = service._get_category_predictions(vid_embeddings, ready_user_params)
            service.alarm_event_manager.update(predictions, ready_sids, ready_user_params)
    torch.cuda.synchronize()

    # Per-step timings; stage totals are derived by summing per iteration.
    # io is intentionally excluded -- input ndarrays are prepared untimed
    # below so the measurement reflects only compute + HTTP.
    steps = {
        "preprocess": [], "model": [], "postprocess": [], "alarm_gate": [],
    }
    per_iter_temp: list[float | None] = []
    t_start = query_gpu_temp_c()

    for _ in range(measure_iters):
        # Untimed input prep -- rotates through the pre-built random pool.
        # Memcpy only; the cost is NOT counted toward any stage total.
        batches = [next_input() for _ in range(batch_size)]

        # Step 1 — preprocess: bgr2rgb (in-place) + roi crop + TF.resize +
        # temporal buffer append + extract ready videos.
        def _pre():
            if not service.is_torch_batches(batches, speed_mode=True):
                cv_bgr2rgb_batch(batches)
            cropped = service.roi_manager.process_batches_with_roi(batches, user_params)
            for sid, frame in zip(stream_ids, cropped):
                service.frame_buffers[sid].append(TF.resize(frame, service.img_size, antialias=True))
            return service._extract_ready_videos(stream_ids, batches, user_params)
        ready, dt = time_call(_pre)
        steps["preprocess"].append(dt * 1000.0)

        if ready is None:
            raise RuntimeError(
                "preprocess produced no ready videos; check TEMPORAL_SIZE / buffer logic"
            )
        batched_videos, ready_sids, ready_user_params, _ = ready

        # Step 2 — model: JPEG encode + HTTP POST /v1/embeddings/batch + parse
        # response. The vision-tower compute lives on the remote container; this
        # process pays for the encode + transport + response parse.
        vid_embeddings, dt = time_call(
            lambda: service._embed_via_server(batched_videos)
        )
        steps["model"].append(dt * 1000.0)

        # Step 3 — postprocess: F.normalize + cos-sim matmul + per-category
        # verdict + user-event filter.
        predictions, dt = time_call(
            lambda: service._get_category_predictions(vid_embeddings, ready_user_params)
        )
        steps["postprocess"].append(dt * 1000.0)

        # Step 4 — alarm_gate: duration queue update + alarm decision.
        _, dt = time_call(
            lambda: service.alarm_event_manager.update(
                predictions, ready_sids, ready_user_params,
            )
        )
        steps["alarm_gate"].append(dt * 1000.0)

        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    # Derive stage totals from step sums (guarantees full >= half >= inference
    # per iteration -- no variance artefacts). io is excluded from every
    # stage (input prep is untimed).
    n = measure_iters
    samples = {
        "full_cycle": [
            steps["preprocess"][i] + steps["model"][i]
            + steps["postprocess"][i] + steps["alarm_gate"][i]
            for i in range(n)
        ],
        "half_cycle": [
            steps["preprocess"][i] + steps["model"][i] + steps["postprocess"][i]
            for i in range(n)
        ],
        "inference": list(steps["model"]),
    }

    stage_stats = {k: stats(v) for k, v in samples.items()}
    step_stats = {k: stats(v) for k, v in steps.items()}
    throughput = {
        f"{k}_imgs_per_s": round(batch_size * 1000.0 / stage_stats[k]["mean_ms"], 2)
        for k in samples
    }
    iterations = {
        "iter": list(range(measure_iters)),
        "preprocess_ms":  [round(v, 3) for v in steps["preprocess"]],
        "model_ms":       [round(v, 3) for v in steps["model"]],
        "postprocess_ms": [round(v, 3) for v in steps["postprocess"]],
        "alarm_gate_ms":  [round(v, 3) for v in steps["alarm_gate"]],
        "full_cycle_ms":  [round(v, 3) for v in samples["full_cycle"]],
        "half_cycle_ms":  [round(v, 3) for v in samples["half_cycle"]],
        "inference_ms":   [round(v, 3) for v in samples["inference"]],
        "gpu_temp_c":     [round(t, 1) if t is not None else None for t in per_iter_temp],
    }

    return {
        "stages": stage_stats,
        "steps": step_stats,
        "throughput": throughput,
        "iterations": iterations,
        "gpu_temperature_c": {
            "start": round(t_start, 1) if t_start is not None else None,
            "end": round(t_end, 1) if t_end is not None else None,
            "min": round(min(temps), 1) if temps else None,
            "max": round(max(temps), 1) if temps else None,
            "mean": round(statistics.fmean(temps), 1) if temps else None,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-VL-Embedding async (remote vLLM server) speed benchmark"
    )
    p.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL,
                   help="vLLM embedding server base URL (default: %(default)s)")
    p.add_argument("--text-features", type=Path, default=DEFAULT_TXT,
                   help="Path to VLE_FP8_text_features.json (text anchors)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="HTTP timeout in seconds for embedding-server calls")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality used when encoding frames to send")
    p.add_argument("--input-h", type=int, default=DEFAULT_INPUT_H,
                   help="Synthetic input height in pixels (default: %(default)s)")
    p.add_argument("--input-w", type=int, default=DEFAULT_INPUT_W,
                   help="Synthetic input width in pixels (default: %(default)s)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--random-pool-size", type=int, default=64,
                   help="Size of the pre-generated random ndarray pool "
                        "(clamped to >= batch*2). Set larger than "
                        "(warmup+iters)*batch to eliminate any cache "
                        "hits from pool wraparound on the remote server.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (local-side preprocess uses GPU tensors)")
    if not args.text_features.exists():
        raise FileNotFoundError(
            f"text_features json not found at {args.text_features} "
            f"(expected VLE_FP8_text_features.json from the Qwen3-VL-Embedding-2B-FP8 release)"
        )

    initial_time = datetime.now()
    info = gpu_info()

    print(f"model:  {QWEN3VLE_ID}  via Qwen3VLEService._detect (remote vLLM server)")
    print(f"server: {args.server_url}")
    print(f"txtfts: {args.text_features}")
    print(f"inputs: synthetic random {args.input_h}x{args.input_w}x{DEFAULT_INPUT_C} uint8 "
          f"(pool={max(args.random_pool_size, args.batch * 2)})")
    print(f"gpu:    {info['name']}  (sm {info['compute_capability']}, "
          f"{info['total_memory_mib']} MiB)")
    print(f"batch:  {args.batch}    warmup: {args.warmup}    iters: {args.iters}")
    print(f"http:   timeout={args.timeout}s  jpeg_quality={args.jpeg_quality}")
    print(f"temporal_size: {TEMPORAL_SIZE}    img_size: {IMG_SIZE}")
    print(f"start:  {initial_time.isoformat(timespec='seconds')}")

    result = benchmark(
        batch_size=args.batch,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        input_h=args.input_h,
        input_w=args.input_w,
        random_pool_size=args.random_pool_size,
    )

    payload = {
        "model": QWEN3VLE_ID,
        "code_path": "Qwen3VLEService._detect (per-step, remote vLLM server)",
        "initial_time": initial_time.isoformat(timespec="seconds"),
        "gpu_type": info["name"],
        "gpu": info,
        "batch_size": args.batch,
        "server_url": args.server_url,
        "text_features_path": str(args.text_features),
        "input_source": "synthetic_random_uint8",
        "input_shape_hwc": [args.input_h, args.input_w, DEFAULT_INPUT_C],
        "random_pool_size": max(args.random_pool_size, args.batch * 2),
        "input_size": [3, *IMG_SIZE],
        "temporal_size": TEMPORAL_SIZE,
        "jpeg_quality": args.jpeg_quality,
        "http_timeout_s": args.timeout,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        **result,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_gpu = info["name"].replace(" ", "_").replace("/", "_")
    ts = initial_time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.out_dir / f"qwen3vle_async_{safe_gpu}_b{args.batch}_{ts}{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print("  steps (per iter, summed -> stage totals; io untimed):")
    for step, s in result["steps"].items():
        print(f"    {step:<12}  mean={s['mean_ms']:>8.3f} ms  "
              f"std={s['std_ms']:>7.3f}  p95={s['p95_ms']:>8.3f} ms")
    print()
    print("  stages (derived: full = pre+model+post+alarm; "
          "half = full-alarm; inference = model):")
    for stage, s in result["stages"].items():
        print(f"    {stage:<12}  mean={s['mean_ms']:>8.3f} ms  "
              f"std={s['std_ms']:>7.3f}  "
              f"p95={s['p95_ms']:>8.3f} ms  "
              f"thr={result['throughput'][f'{stage}_imgs_per_s']:>7.1f} img/s")
    print(f"\nGPU temp: {result['gpu_temperature_c']['start']} -> "
          f"{result['gpu_temperature_c']['end']} C  "
          f"(min={result['gpu_temperature_c']['min']}, "
          f"max={result['gpu_temperature_c']['max']})")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
