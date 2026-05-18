"""Speed benchmark for perception_encoder via the real PEService pipeline.

Instantiates ``pia_prod.AI.modules.perception_encoder.service.PEService`` and
times the three split stages exposed in ``_detect`` -- the same code path
production runs.

Stage boundary convention (shared with the FT_PE bench):
    * ``half_cycle`` is the **video-side** pipeline -- everything up to and
      including the latest per-stream **video embedding** ``(B, 1024)``. PE
      has no temporal model and ``TEMPORAL_SIZE = 1``, so the per-stream
      video embedding is the per-image visual vector itself.
    * The moment any text-side work runs (text embeddings, cos-sim against
      text features, top-K, alarm event manager), the timing is no longer
      ``half_cycle`` -- that work lives only in ``full_cycle``.

    full_cycle:  disk read -> _preprocess_stage -> _inference_stage
                 -> _postprocess_stage
                 (end-to-end: deque append -> cos sim vs text features
                  -> top-K -> duration-queue alarm)
    half_cycle:  in-memory ndarray -> _preprocess_stage -> _inference_stage
                 (stops at video emb (B, 1024); no text-side work)
    inference:   already-preprocessed CUDA tensor -> _inference_stage
                 (stops at visual emb (B, 1024); no preprocess, no text)

GPU temperature is polled per iter via NVML when ``pynvml`` is installed,
falling back to nvidia-smi otherwise.
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "Product-AI-mono" / "packages"))

DEFAULT_ENGINE = REPO_ROOT / "assets" / "model" / "PE-Core-L14-336_vision_dynamic.engine"
DEFAULT_TXT = REPO_ROOT / "assets" / "model" / "text_features.json"
DEFAULT_IMAGE = REPO_ROOT / "assets" / "images" / "kkpolice_1.jpg"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# ---- Env-var injection BEFORE importing PEService ---------------------------
# pia_prod.AI.modules.perception_encoder.config reads these at import time, so
# we have to set them before the import chain starts.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
_pre.add_argument("--text-features", type=Path, default=DEFAULT_TXT)
_pre_args, _ = _pre.parse_known_args()
os.environ["MODEL_PERCEPTION_ENCODER_TRT_PATH"] = str(_pre_args.engine)
os.environ["MODEL_PERCEPTION_ENCODER_TXT_FEATURE_PATH"] = str(_pre_args.text_features)
# -----------------------------------------------------------------------------

from queue import Queue  # noqa: E402

from pia.ai.tasks.T2VRet.models.PE.utils.complexity_check import (  # noqa: E402
    time_call,
    get_gpu_stats_nvml,
)
from pia_prod.AI.modules.perception_encoder.service import PEService  # noqa: E402
from pia_prod.AI.modules.perception_encoder.config import IMG_SIZE  # noqa: E402


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


def load_image_ndarray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), copy=True)


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
    """user_param payloads for PERoIManager + PEEventManager with fire / falldown
    / smoke retEvents enabled.

    Shape mirrors what ``AddStreamModel2dict`` produces in production: retEvent
    is a dict keyed by category id, each value carrying a ``roi`` with empty
    ``polygonCoordinates`` so the PE ROI manager falls back to the whole frame.
    Falldown is included on purpose -- it's the only key in
    ``PERoIManager.roi_category_list``, so this exercises the dict-shape ROI
    lookup path that production hits."""
    return [
        {"user_param": {
            "retEvent": {
                "fire_ret":     {"roi": {"polygonCoordinates": []}},
                "falldown_ret": {"roi": {"polygonCoordinates": []}},
                "smoke_ret":    {"roi": {"polygonCoordinates": []}},
            },
            "cameraId": f"cam_{i}",
            "organization": "pia",
        }}
        for i in range(batch_size)
    ]


@torch.inference_mode()
def benchmark(
    image_path: Path,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    service = PEService(Queue())
    user_params = make_user_params(batch_size)

    # cv_bgr2rgb_batch (called inside _preprocess_stage) mutates ndarrays in
    # place, so each timed call needs fresh copies of the frame.
    template = load_image_ndarray(image_path)

    def fresh_batches() -> list[np.ndarray]:
        return [template.copy() for _ in range(batch_size)]

    stream_ids = [f"stream_{i}" for i in range(batch_size)]

    # One preprocessed CUDA tensor reused across iters for the inference-only
    # stage.
    preprocessed = service._preprocess_stage(fresh_batches(), user_params)

    # Warmup covers the full _detect chain so the first measured full_cycle
    # iter doesn't pay the postprocess (cos-sim + top-K + alarm event manager)
    # cold-cache hit.
    for _ in range(warmup_iters):
        batches = fresh_batches()
        x = service._preprocess_stage(batches, user_params)
        v = service._inference_stage(x)
        _ = service._postprocess_stage(v, batches, stream_ids, user_params)
    torch.cuda.synchronize()

    samples = {"full_cycle": [], "half_cycle": [], "inference": []}
    per_iter_temp: list[float | None] = []
    t_start = query_gpu_temp_c()

    in_mem = fresh_batches()  # the "ndarray already in RAM" baseline buffer

    for _ in range(measure_iters):
        # full_cycle: disk read -> preprocess -> inference -> postprocess.
        # End-to-end: also runs the cos-sim + top-K + alarm event manager.
        def _full():
            batches = [load_image_ndarray(image_path) for _ in range(batch_size)]
            x = service._preprocess_stage(batches, user_params)
            v = service._inference_stage(x)
            return service._postprocess_stage(v, batches, stream_ids, user_params)
        _, dt = time_call(_full)
        samples["full_cycle"].append(dt * 1000.0)

        # half_cycle: in-memory ndarray -> preprocess -> inference (stops at img emb).
        # The .copy() simulates production receiving a fresh frame each tick.
        def _half():
            batches = [b.copy() for b in in_mem]
            x = service._preprocess_stage(batches, user_params)
            return service._inference_stage(x)
        _, dt = time_call(_half)
        samples["half_cycle"].append(dt * 1000.0)

        # inference: preprocessed CUDA tensor -> inference (stops at img emb).
        _, dt = time_call(lambda: service._inference_stage(preprocessed))
        samples["inference"].append(dt * 1000.0)

        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    stage_stats = {k: stats(v) for k, v in samples.items()}
    throughput = {
        f"{k}_imgs_per_s": round(batch_size * 1000.0 / stage_stats[k]["mean_ms"], 2)
        for k in samples
    }
    iterations = {
        "iter": list(range(measure_iters)),
        "full_cycle_ms": [round(v, 3) for v in samples["full_cycle"]],
        "half_cycle_ms": [round(v, 3) for v in samples["half_cycle"]],
        "inference_ms":  [round(v, 3) for v in samples["inference"]],
        "gpu_temp_c":    [round(t, 1) if t is not None else None for t in per_iter_temp],
    }

    return {
        "stages": stage_stats,
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
        description="PE TRT speed benchmark (via PEService._detect stages)"
    )
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                   help="TensorRT engine path (.trt / .engine)")
    p.add_argument("--text-features", type=Path, default=DEFAULT_TXT,
                   help="Path to text_features.json from PIA-SPACE-LAB/PE-Core-L14-336")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.engine.suffix not in (".trt", ".engine"):
        raise ValueError(
            f"engine must be .trt or .engine, got {args.engine.suffix}: {args.engine}"
        )
    if not args.text_features.exists():
        raise FileNotFoundError(
            f"text_features.json not found at {args.text_features} "
            f"(download from PIA-SPACE-LAB/PE-Core-L14-336)"
        )

    initial_time = datetime.now()
    info = gpu_info()

    print(f"model:  PE (perception_encoder)  via PEService._detect")
    print(f"engine: {args.engine}")
    print(f"txtfts: {args.text_features}")
    print(f"image:  {args.image}")
    print(f"gpu:    {info['name']}  (sm {info['compute_capability']}, "
          f"{info['total_memory_mib']} MiB)")
    print(f"batch:  {args.batch}    warmup: {args.warmup}    iters: {args.iters}")
    print(f"start:  {initial_time.isoformat(timespec='seconds')}")

    result = benchmark(
        image_path=args.image,
        batch_size=args.batch,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
    )

    payload = {
        "model": "PE-Core-L14-336",
        "code_path": "PEService._detect (split stages)",
        "initial_time": initial_time.isoformat(timespec="seconds"),
        "gpu_type": info["name"],
        "gpu": info,
        "batch_size": args.batch,
        "engine_path": str(args.engine),
        "text_features_path": str(args.text_features),
        "image_path": str(args.image),
        "input_size": [3, *IMG_SIZE],
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
    out_path = args.out_dir / f"pe_{safe_gpu}_b{args.batch}_{ts}{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    for stage, s in result["stages"].items():
        print(f"  {stage:<11}  mean={s['mean_ms']:>8.3f} ms  "
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
