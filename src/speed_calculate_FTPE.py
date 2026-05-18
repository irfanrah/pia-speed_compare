"""Speed benchmark for ft_pe via the real FTPEService pipeline.

Instantiates ``pia_prod.AI.modules.ft_pe.service.FTPEService`` and times
four split stages exposed in ``_detect``.

Stage boundary convention (shared with the PE bench):
    * ``half_cycle`` = **one production tick of the vision side**: disk read
      + ``_preprocess_stage`` + gather buffer + temporal model + frame buffer
      + per-stream mean-pool -> video embedding ``(B, 1024)``.
    * ``full_cycle`` = same tick **plus** the text side: L2 norm + per-class
      cos-sim against text features + alarm event manager.
    * ``three_quarters_cycle`` = ``full_cycle`` minus the disk read; frames
      already in RAM (e.g. handed in by a camera/grabber buffer) when the
      timed call starts.
    * So ``half_cycle <= full_cycle`` by construction (only the text side
      differs); ``three_quarters_cycle <= full_cycle`` likewise (only the
      disk read differs).

    full_cycle:            disk read -> _preprocess_stage -> _postprocess_stage
                           (end-to-end per-tick production cycle: gather
                            buffer + sliding-window encode + frame buffer
                            + mean-pool + L2 norm + per-class cos sim +
                            alarm event manager)
    three_quarters_cycle:  in-memory ndarray -> _preprocess_stage
                           -> _postprocess_stage
                           (full_cycle minus disk read)
    half_cycle:            disk read -> _preprocess_stage(B)
                           -> (gather buffer + model + frame buffer +
                            per-stream mean-pool, reusing service state)
                           (stops at video_emb (B, 1024); no text-side work)
    inference:             already-preprocessed CUDA tensor (B,T,C,H,W)
                           -> _inference_stage
                           (stops at per-frame img emb (B, T, 1024); no
                            preprocess, no mean-pool, no text)

half_cycle and three_quarters_cycle both reuse
``service.gather_frame_buffers`` and ``service.frame_buffers``, so each
call advances the per-stream state by exactly one tick -- the same advance
``_detect`` causes for full_cycle. The timed loop runs full / three-quarters
/ half / inference within each iter; the first three each advance the state
once, inference does not (the buffer stays full across all advances).

Throughput is reported in **frames encoded per second**
(``*_imgs_per_s = B * T * 1000 / mean_ms``). Production input rate (streams
per second, i.e. unique frames ingested per tick) is also reported for the
three tick-based stages as ``throughput.<stage>_streams_per_s``.

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


def make_user_params(count: int, cam_prefix: str = "cam") -> list[dict]:
    """user_param payloads with fire / falldown / smoke FT_PE retEvents on.

    Shape mirrors what ``AddStreamModel2dict`` produces in production: retEvent
    is a dict keyed by category id, each value carrying a ``roi`` with empty
    ``polygonCoordinates`` so FTPERoIManager (whose ``roi_category_list`` is
    ``ALL_CATEGORIES``) takes the dict-lookup path AND falls back to a whole-
    frame ROI."""
    return [
        {"user_param": {
            "retEvent": {
                "fire_ft_ret":     {"roi": {"polygonCoordinates": []}},
                "falldown_ft_ret": {"roi": {"polygonCoordinates": []}},
                "smoke_ft_ret":    {"roi": {"polygonCoordinates": []}},
            },
            "cameraId": f"{cam_prefix}_{i}",
            "organization": "pia",
        }}
        for i in range(count)
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
    # stage by replicating one tick's preprocessed frames across the temporal
    # axis. The TRT model's compute is independent of pixel values, so timing
    # is identical to production (where T frames in the window are distinct).
    # Engine I/O is FP32 -- do NOT cast to .half().
    _single_tick = service._preprocess_stage(fresh_batches(), user_params)  # (B, 3, H, W)
    preprocessed_bt = (
        _single_tick.unsqueeze(1).expand(-1, frames, -1, -1, -1).contiguous()
    )                                                                       # (B, T, 3, H, W)

    # --- Mirror of FTPEService._postprocess_stage up to (B, 1024) video_emb,
    # omitting the text-side work (L2 norm + per-class cos-sim + alarm event
    # manager). Reuses ``service.gather_frame_buffers`` and
    # ``service.frame_buffers``, so each call advances the per-stream state
    # by exactly one tick -- the same advance full_cycle's ``_detect`` causes.
    def _postprocess_to_video_emb(processed_batches, stream_ids):
        for stream_id, batch in zip(stream_ids, processed_batches):
            service.gather_frame_buffers[stream_id].append(batch)
            service._unconsumed_frames[stream_id] += 1

        encode_stream_ids = []
        encode_tensors = []
        for stream_id in stream_ids:
            gbuf = service.gather_frame_buffers[stream_id]
            if (
                len(gbuf) == service.window_size
                and service._unconsumed_frames[stream_id] >= service.stride
            ):
                encode_tensors.append(torch.stack(list(gbuf)))            # (W, C, H, W)
                service._unconsumed_frames[stream_id] -= service.stride
                encode_stream_ids.append(stream_id)

        if encode_stream_ids:
            encode_input = torch.stack(encode_tensors)                    # (B_enc, W, ...)
            embeddings = service._inference_stage(encode_input)            # (B_enc, W, 1024)
            for sid, emb in zip(encode_stream_ids, embeddings):
                new_embs = emb[-service.prediction_size:]
                service.frame_buffers[sid].extend(new_embs.unbind(0))
                if len(service.frame_buffers[sid]) > TEMPORAL_SIZE:
                    del service.frame_buffers[sid][1]

        video_embeddings = []
        for stream_id in stream_ids:
            buf = service.frame_buffers[stream_id]
            if len(buf) >= TEMPORAL_SIZE:
                stacked = torch.stack(list(buf))                          # (TEMPORAL_SIZE, 1024)
                video_embeddings.append(stacked.mean(dim=0))              # (1024,)

        if not video_embeddings:
            return None
        return torch.stack(video_embeddings)                              # (B, 1024)

    # --- Prime the FTPEService temporal buffer so each measured tick (full
    # OR half) actually produces a video embedding.
    buffer_warmup = TEMPORAL_SIZE + service.window_size + 2
    for _ in range(buffer_warmup):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )

    # Additional warmup for cache/JIT. ``_detect`` covers half_cycle's
    # vision-side path (preprocess + buffer + model + frame_buffer + mean),
    # so full_cycle warmup is enough for both. The inference-only stage gets
    # its own warmup hit.
    for _ in range(warmup_iters):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )
        _ = service._inference_stage(preprocessed_bt)
    torch.cuda.synchronize()

    samples = {
        "full_cycle": [],
        "three_quarters_cycle": [],
        "half_cycle": [],
        "inference": [],
    }
    per_iter_temp: list[float | None] = []
    t_start = query_gpu_temp_c()

    in_mem = fresh_batches()  # "ndarray already in RAM" baseline for three_quarters

    for _ in range(measure_iters):
        # full_cycle: one production tick. disk read -> _detect (preprocess
        # + gather/frame buffers + model + per-stream mean + L2 norm + per-
        # category cos-sim vs text features + alarm event manager). Both
        # buffers are primed so every call yields an alarm decision.
        def _full():
            batches = [load_image_ndarray(image_path) for _ in range(batch_size)]
            return service._detect(
                batches=batches, stream_ids=stream_ids, user_params=user_params,
            )
        _, dt = time_call(_full)
        samples["full_cycle"].append(dt * 1000.0)

        # three_quarters_cycle: same tick as full_cycle MINUS the disk read.
        # Frames already in RAM (e.g. handed in by a camera/grabber buffer).
        def _three_quarters():
            batches = [b.copy() for b in in_mem]
            return service._detect(
                batches=batches, stream_ids=stream_ids, user_params=user_params,
            )
        _, dt = time_call(_three_quarters)
        samples["three_quarters_cycle"].append(dt * 1000.0)

        # half_cycle: same tick as full_cycle MINUS the text-side work.
        # disk read -> _preprocess_stage(B) -> postprocess up to per-stream
        # mean-pool = (B, 1024) video embeddings. Strictly equal to one
        # full_cycle tick minus L2 norm + cos-sim + alarm, so by construction
        # half_cycle <= full_cycle.
        def _half():
            batches = [load_image_ndarray(image_path) for _ in range(batch_size)]
            x = service._preprocess_stage(batches, user_params)
            return _postprocess_to_video_emb(x, stream_ids)
        _, dt = time_call(_half)
        samples["half_cycle"].append(dt * 1000.0)

        # inference: pre-cooked CUDA tensor -> model only.
        _, dt = time_call(lambda: service._inference_stage(preprocessed_bt))
        samples["inference"].append(dt * 1000.0)

        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    stage_stats = {k: stats(v) for k, v in samples.items()}
    # Unit: frames encoded per second (B*T per tick / per call).
    #   - full / three_quarters / half_cycle: stride=1 with window_size=T
    #     means each tick re-encodes a (B, T) window = B*T frames per tick.
    #   - inference: one call encodes a (B, T) tensor, i.e. B*T frames.
    # All four are directly comparable. Per-tick *input* rate (production
    # streams per second) is reported separately for the tick-based stages.
    total_frames = batch_size * frames
    throughput = {
        f"{k}_imgs_per_s": round(total_frames * 1000.0 / stage_stats[k]["mean_ms"], 2)
        for k in samples
    }
    for k in ("full_cycle", "three_quarters_cycle", "half_cycle"):
        throughput[f"{k}_streams_per_s"] = round(
            batch_size * 1000.0 / stage_stats[k]["mean_ms"], 2
        )

    iterations = {
        "iter": list(range(measure_iters)),
        "full_cycle_ms":           [round(v, 3) for v in samples["full_cycle"]],
        "three_quarters_cycle_ms": [round(v, 3) for v in samples["three_quarters_cycle"]],
        "half_cycle_ms":           [round(v, 3) for v in samples["half_cycle"]],
        "inference_ms":            [round(v, 3) for v in samples["inference"]],
        "gpu_temp_c":              [round(t, 1) if t is not None else None for t in per_iter_temp],
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
