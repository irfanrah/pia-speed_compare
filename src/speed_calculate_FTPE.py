"""Speed benchmark for ft_pe (fine-tuned PE) TRT model with temporal dim.

Loads ``PiaONNXTensorRTModel`` from ``pia.ai.model`` and the shared
``preprocess_image`` from ``pia_prod.AI.modules.perception_encoder.trt_utils``
- the same pipeline ``pia_prod.AI.modules.ft_pe.service.FTPEService`` uses.

Input shape: (B, T, 3, 336, 336). T (window/temporal frames) defaults to the
engine's optimization point (8); B defaults to 8.

Measures three stages per iteration:

    full_cycle:  disk read -> preprocess -> inference   (end-to-end)
    half_cycle:  in-memory ndarray -> preprocess -> inference
    inference:   already-preprocessed CUDA tensor -> inference
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "Product-AI-mono" / "packages"))

from pia.ai.model import PiaONNXTensorRTModel  # noqa: E402
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image  # noqa: E402

DEFAULT_ENGINE = (
    REPO_ROOT / "assets" / "model"
    / "FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine"
)
DEFAULT_IMAGE = REPO_ROOT / "assets" / "images" / "kkpolice_1.jpg"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

IMG_SIZE = (336, 336)


def query_gpu_temp_c(device_index: int = 0) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
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


def _to_btchw(preprocessed_bchw: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
    """Reshape (B*T, 3, H, W) -> (B, T, 3, H, W) for the FT_PE temporal model."""
    assert preprocessed_bchw.shape[0] == batch * frames, (
        preprocessed_bchw.shape, batch, frames
    )
    return preprocessed_bchw.view(batch, frames, *preprocessed_bchw.shape[1:]).contiguous()


@torch.inference_mode()
def benchmark(
    engine_path: Path,
    image_path: Path,
    batch_size: int,
    frames: int,
    warmup_iters: int,
    measure_iters: int,
    half: bool,
) -> dict:
    if not engine_path.exists():
        raise FileNotFoundError(f"engine not found: {engine_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    model = PiaONNXTensorRTModel(str(engine_path), device="cuda", half=half)

    total_frames = batch_size * frames
    frame_ndarray = load_image_ndarray(image_path)
    in_mem_batch = [frame_ndarray] * total_frames
    preprocessed = _to_btchw(
        preprocess_image(in_mem_batch, size=IMG_SIZE[0], device="cuda"),
        batch_size, frames,
    )
    if half:
        preprocessed = preprocessed.half()

    for _ in range(warmup_iters):
        x = preprocess_image(in_mem_batch, size=IMG_SIZE[0], device="cuda")
        x = _to_btchw(x, batch_size, frames)
        if half:
            x = x.half()
        _ = model(x)
    torch.cuda.synchronize()

    samples = {"full_cycle": [], "half_cycle": [], "inference": []}
    per_iter_temp: list[float | None] = []

    t_start = query_gpu_temp_c()

    for _ in range(measure_iters):
        # full_cycle: disk read -> preprocess -> inference
        t0 = time.perf_counter()
        frame = load_image_ndarray(image_path)
        batch = [frame] * total_frames
        x = preprocess_image(batch, size=IMG_SIZE[0], device="cuda")
        x = _to_btchw(x, batch_size, frames)
        if half:
            x = x.half()
        _ = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples["full_cycle"].append((t1 - t0) * 1000.0)

        # half_cycle: ndarray in memory -> preprocess -> inference
        t0 = time.perf_counter()
        x = preprocess_image(in_mem_batch, size=IMG_SIZE[0], device="cuda")
        x = _to_btchw(x, batch_size, frames)
        if half:
            x = x.half()
        _ = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples["half_cycle"].append((t1 - t0) * 1000.0)

        # inference: preprocessed CUDA tensor -> inference
        t0 = time.perf_counter()
        _ = model(preprocessed)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples["inference"].append((t1 - t0) * 1000.0)

        # GPU temp sampled AFTER the three timed stages so the nvidia-smi
        # subprocess overhead doesn't pollute the timing.
        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    stage_stats = {k: stats(v) for k, v in samples.items()}
    throughput = {
        f"{k}_imgs_per_s": round(total_frames * 1000.0 / stage_stats[k]["mean_ms"], 2)
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
    p = argparse.ArgumentParser(description="FT_PE TRT speed benchmark")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                   help="TensorRT engine path (.trt / .engine)")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--frames", type=int, default=1,
                   help="Temporal window size T in (B, T, 3, H, W). "
                        "1 matches FPS_8 mode, 3 matches FPS_3 mode in ft_pe.service.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--no-half", action="store_true",
                   help="Disable FP16 input (default FP16, matching service.py)")
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

    half = not args.no_half
    initial_time = datetime.now()
    info = gpu_info()

    print(f"model:  FT_PE (ft_pe)")
    print(f"engine: {args.engine}")
    print(f"image:  {args.image}")
    print(f"gpu:    {info['name']}  (sm {info['compute_capability']}, "
          f"{info['total_memory_mib']} MiB)")
    print(f"batch:  {args.batch}  frames: {args.frames}  half: {half}  "
          f"warmup: {args.warmup}  iters: {args.iters}")
    print(f"start:  {initial_time.isoformat(timespec='seconds')}")

    result = benchmark(
        engine_path=args.engine,
        image_path=args.image,
        batch_size=args.batch,
        frames=args.frames,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        half=half,
    )

    payload = {
        "model": "FT_PE-Core-L14-336",
        "initial_time": initial_time.isoformat(timespec="seconds"),
        "gpu_type": info["name"],
        "gpu": info,
        "batch_size": args.batch,
        "frames": args.frames,
        "total_frames_per_iter": args.batch * args.frames,
        "engine_path": str(args.engine),
        "image_path": str(args.image),
        "input_size": [args.frames, 3, *IMG_SIZE],
        "precision": "fp16" if half else "fp32",
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
    out_path = (
        args.out_dir
        / f"ftpe_{safe_gpu}_b{args.batch}_t{args.frames}_{ts}{suffix}.json"
    )
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
