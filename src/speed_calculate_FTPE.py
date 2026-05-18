"""Speed benchmark for ft_pe via the real FTPEService pipeline.

Instantiates ``pia_prod.AI.modules.ft_pe.service.FTPEService`` and times the
three split stages exposed in ``_detect``:

    full_cycle:  disk read -> _preprocess_stage -> _postprocess_stage
                 (end-to-end per-tick production cycle: includes sliding-
                 window encode + cos sim + alarm)
    half_cycle:  in-memory ndarrays for B*T frames -> _preprocess_stage
                 -> _inference_stage(reshape (B*T,C,H,W) -> (B,T,C,H,W))
                 (synthetic bulk run, stops at img emb)
    inference:   already-preprocessed CUDA tensor (B,T,C,H,W)
                 -> _inference_stage   (stops at img emb)

The full_cycle warms up FTPEService's temporal buffer for ``TEMPORAL_SIZE +
window_size`` ticks before the timed loop so every measured iter produces a
prediction. Defaults assume the ``8fps`` mode (window_size=1, matches an
engine built with --max-frames=1).
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

DEFAULT_ENGINE = (
    REPO_ROOT / "assets" / "model"
    / "FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine"
)
DEFAULT_TXT = REPO_ROOT / "assets" / "model" / "FT_text_features.json"
DEFAULT_IMAGE = REPO_ROOT / "assets" / "images" / "kkpolice_1.jpg"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# ---- Env-var injection BEFORE importing FTPEService -------------------------
# ft_pe.config reads these at import time, so we have to set them first.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
_pre.add_argument("--text-features", type=Path, default=DEFAULT_TXT)
_pre_args, _ = _pre.parse_known_args()
os.environ["MODEL_PE_VIOLENCE_DETECTION_TRT_PATH"] = str(_pre_args.engine)
os.environ["FT_PE_TEXT_FEATURES_JSON"] = str(_pre_args.text_features)
# Pick the lightest FT_PE_MODE for config import; window_size is overridden
# after instantiation to match --frames anyway.
os.environ.setdefault("FT_PE_MODE", "8fps")
# -----------------------------------------------------------------------------

from queue import Queue  # noqa: E402

from pia.ai.tasks.T2VRet.models.PE.utils.complexity_check import (  # noqa: E402
    time_call,
    get_gpu_stats_nvml,
)
from pia_prod.AI.modules.ft_pe.service import FTPEService  # noqa: E402
from pia_prod.AI.modules.ft_pe.config import IMG_SIZE, TEMPORAL_SIZE  # noqa: E402
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image  # noqa: E402


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
    """retEvent uses a non-matching key so the ROI manager defaults to the
    whole frame (FTPERoIManager treats every ALL_CATEGORIES name as ROI-bearing
    and requires a dict shape — we want to avoid that path)."""
    return [
        {"user_param": {
            "retEvent": ["speed_test_ret"],
            "cameraId": f"cam_{i}",
            "organization": "pia",
        }}
        for i in range(batch_size)
    ]


@torch.inference_mode()
def benchmark(
    image_path: Path,
    batch_size: int,
    frames: int,
    warmup_iters: int,
    measure_iters: int,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    service = FTPEService(Queue())

    # Pre-flight: verify (batch_size, frames) lies within the engine's
    # optimization profile, otherwise the model call later crashes with a
    # cryptic CUDA illegal-memory-access cascade.
    # PiaONNXTensorRTModel.load_tensorrt_model does
    # ``self.__dict__.update(locals())`` so the deserialized engine ends up
    # at ``service.model.model`` (the local was named ``model``).
    eng = service.model.model
    in_name = service.model.input_names[0]
    mn, _opt, mx = eng.get_tensor_profile_shape(in_name, 0)
    if not (mn[0] <= batch_size <= mx[0]) or not (mn[1] <= frames <= mx[1]):
        raise ValueError(
            f"requested shape (B={batch_size}, T={frames}) is outside the "
            f"engine's profile {tuple(mn)} .. {tuple(mx)}. "
            f"Rebuild with MAX_BATCH/MAX_FRAMES big enough (delete the .engine first)."
        )

    # Override sliding-window state so --frames T drives the full_cycle's
    # encode shape. With sliding_window_size = frames-1, stride=1 so every
    # tick triggers an encode + decision.
    service.window_size = frames
    service.sliding_window_size = max(0, frames - 1)
    service.prediction_size = 1
    service.stride = max(1, service.window_size - service.sliding_window_size)
    # gather_frame_buffers is a defaultdict; the lambda captures self.window_size
    # by closure so future deques pick up the new maxlen. No streams have been
    # registered yet, so no existing deques to migrate.

    stream_ids = [f"stream_{i}" for i in range(batch_size)]
    user_params = make_user_params(batch_size)
    template = load_image_ndarray(image_path)

    def fresh_batches() -> list[np.ndarray]:
        return [template.copy() for _ in range(batch_size)]

    # --- Build a preprocessed (B, T, C, H, W) tensor for the inference-only
    # stage (synthetic flow that bypasses the per-stream buffer).
    # Engine I/O is FP32 (see engine inspection) -- do NOT cast to .half().
    bulk = [template.copy() for _ in range(batch_size * frames)]
    bulk_cuda = preprocess_image(bulk, size=IMG_SIZE[0], device="cuda")
    preprocessed_bt = bulk_cuda.view(
        batch_size, frames, *bulk_cuda.shape[1:]
    ).contiguous()

    # --- Prime the FTPEService temporal buffer so each measured full_cycle
    # tick actually produces a prediction.
    buffer_warmup = TEMPORAL_SIZE + service.window_size + 2
    for _ in range(buffer_warmup):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )

    # Additional warmup for cache/JIT.
    for _ in range(warmup_iters):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )
        _ = service._inference_stage(preprocessed_bt)
    torch.cuda.synchronize()

    samples = {"full_cycle": [], "half_cycle": [], "inference": []}
    per_iter_temp: list[float | None] = []
    t_start = query_gpu_temp_c()

    in_mem_bulk = [template.copy() for _ in range(batch_size * frames)]

    for _ in range(measure_iters):
        # full_cycle: disk read -> _detect (preprocess + buffer + encode +
        # decide). Buffer is already primed so every call yields a decision.
        def _full():
            batches = [load_image_ndarray(image_path) for _ in range(batch_size)]
            return service._detect(
                batches=batches, stream_ids=stream_ids, user_params=user_params,
            )
        _, dt = time_call(_full)
        samples["full_cycle"].append(dt * 1000.0)

        # half_cycle: in-memory B*T frames -> preprocess -> inference (stops
        # at img emb). Synthetic bulk path -- not the production per-tick
        # buffer flow.
        def _half():
            bulk_now = [b.copy() for b in in_mem_bulk]
            x = preprocess_image(bulk_now, size=IMG_SIZE[0], device="cuda")
            x = x.view(batch_size, frames, *x.shape[1:]).contiguous()
            return service._inference_stage(x)
        _, dt = time_call(_half)
        samples["half_cycle"].append(dt * 1000.0)

        # inference: preprocessed CUDA tensor -> inference (stops at img emb).
        _, dt = time_call(lambda: service._inference_stage(preprocessed_bt))
        samples["inference"].append(dt * 1000.0)

        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    stage_stats = {k: stats(v) for k, v in samples.items()}
    total_frames = batch_size * frames
    throughput = {
        f"{k}_imgs_per_s": round(total_frames * 1000.0 / stage_stats[k]["mean_ms"], 2)
        for k in samples
    }
    # full_cycle processes 1 frame per stream per tick, so it has a different
    # work unit -- override its throughput to reflect that.
    throughput["full_cycle_imgs_per_s"] = round(
        batch_size * 1000.0 / stage_stats["full_cycle"]["mean_ms"], 2
    )

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
        description="FT_PE TRT speed benchmark (via FTPEService._detect stages)"
    )
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                   help="TensorRT engine path (.trt / .engine)")
    p.add_argument("--text-features", type=Path, default=DEFAULT_TXT)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--frames", type=int, default=1,
                   help="Temporal window T (any value within the engine's profile, "
                        "typically 1, 3, or 8). Drives both the synthetic inference "
                        "T and FTPEService.window_size for full_cycle.")
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
            f"FT_text_features.json not found at {args.text_features}"
        )

    initial_time = datetime.now()
    info = gpu_info()

    print(f"model:  FT_PE (ft_pe)  via FTPEService._detect")
    print(f"engine: {args.engine}")
    print(f"txtfts: {args.text_features}")
    print(f"T:      {args.frames}  (window_size driven by --frames)")
    print(f"image:  {args.image}")
    print(f"gpu:    {info['name']}  (sm {info['compute_capability']}, "
          f"{info['total_memory_mib']} MiB)")
    print(f"batch:  {args.batch}  frames: {args.frames}  "
          f"warmup: {args.warmup}  iters: {args.iters}")
    print(f"start:  {initial_time.isoformat(timespec='seconds')}")

    result = benchmark(
        image_path=args.image,
        batch_size=args.batch,
        frames=args.frames,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
    )

    payload = {
        "model": "FT_PE-Core-L14-336",
        "code_path": "FTPEService._detect (split stages)",
        "initial_time": initial_time.isoformat(timespec="seconds"),
        "gpu_type": info["name"],
        "gpu": info,
        "batch_size": args.batch,
        "frames": args.frames,
        "engine_path": str(args.engine),
        "text_features_path": str(args.text_features),
        "image_path": str(args.image),
        "input_size": [args.frames, 3, *IMG_SIZE],
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
