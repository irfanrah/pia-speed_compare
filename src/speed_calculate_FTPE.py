"""Speed benchmark for ft_pe via the real FTPEService pipeline.

Instantiates ``pia_prod.AI.modules.ft_pe.service.FTPEService`` and times
five split stages.

Stage boundary convention (shared with the PE bench):
    * ``half_cycle`` = vision-side per-tick work starting at
      ``_preprocess_stage`` and stopping at the per-stream video embedding
      ``(B, 1024)``.
    * ``full_cycle`` = same start point (preprocess) but continues through
      the text side: L2 norm + per-class cos-sim against text features +
      alarm event manager.
    * ``full_cycle`` and ``half_cycle`` are sampled from the **same**
      shared run inside each iter: the timer is snapshotted at the half
      boundary, the run continues into the text-side block, and the timer
      is snapshotted again for full. So ``half_cycle <= full_cycle`` by
      construction and ``full - half`` is exactly the text-side cost,
      free of preprocess+inference jitter.

    full_cycle:           _preprocess_stage -> _postprocess_stage
                          (end-to-end per-tick from preprocess: gather
                           buffer + sliding-window encode + frame buffer
                           + mean-pool + L2 norm + per-class cos sim +
                           alarm event manager)
    half_cycle:           _preprocess_stage(B) -> (gather buffer + model +
                          frame buffer + per-stream mean-pool, reusing
                          service state)
                          (stops at video_emb (B, 1024); no text-side work)
    inference:            already-preprocessed CUDA tensor (B,T,C,H,W)
                          -> _inference_stage
                          (stops at per-frame img emb (B, T, 1024); no
                           preprocess, no mean-pool, no text)
    input_gen_and_load:   isolated cost of B random 1080p uint8 ndarrays
                          ("load from disk / camera buffer" stand-in)
    cos_sim:              isolated per-class cos-sim matmul, per category:
                          (vis @ cat_txt[c]).max(1) and (vis @ cat_normal[c]).max(1).
                          Stops at the dot product -- no L2 norm of
                          video_embs, no `>` comparison + .cpu().tolist()
                          sync, no alarm_event_manager.update.

The timed region starts at ``_preprocess_stage`` with batches drawn from
a **pre-generated input pool** (one big list of unique random
(1080, 1920, 3) uint8 ndarrays materialized once at bench start, sized to
cover every B-batch consumer: buffer_warmup + cache warmup + measure
loop). ``fresh_batches()`` returns the next B slots from the pool,
.copy()-ing so cv_bgr2rgb_batch can mutate the consumer without trashing
the source. Each fresh_batches() call advances the pool index, so every
iter sees disjoint random data. ``full_cycle``, ``half_cycle``,
``inference``, and ``cos_sim`` all share the **same** fresh_batches()
call within an iter: one preprocess+model+mean run feeds the shared
full/half snapshots, then ``inference`` re-times the model call on a
(B, T) tensor built from the same ``x`` and ``cos_sim`` re-times the
matmul on the same ``vis_vectors`` produced by the shared run.
Input-prep cost is reported separately as ``input_gen_and_load`` (timed
as a fresh ``gen_random_frame() × B``, NOT from the pool -- it
deliberately measures the cost of producing new data). The ``--image``
CLI flag is preserved for backwards compat but ignored.

``full_cycle``/``half_cycle`` reuse ``service.gather_frame_buffers`` and
``service.frame_buffers``, advancing the per-stream state by exactly one
tick per measure iter (one shared run, not two).

Throughput is reported in **frames encoded per second**
(``*_imgs_per_s = B * T * 1000 / mean_ms``). Production input rate (streams
per second, i.e. unique frames ingested per tick) is also reported for the
tick-based stages as ``throughput.<stage>_streams_per_s``.

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
import time
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
    cuda_sync,
    time_call,
    get_gpu_stats_nvml,
)
from pia_prod.AI.modules.ft_pe.service import FTPEService  # noqa: E402
from pia_prod.AI.modules.ft_pe.config import (  # noqa: E402
    ABNORMAL_CLASS_NAMES,
    IMG_SIZE,
    TEMPORAL_SIZE,
)


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


_FRAME_RNG = np.random.default_rng()
_FRAME_HW = (1080, 1920, 3)  # match a typical 1080p RGB camera frame


def gen_random_frame() -> np.ndarray:
    """Generate a fresh (1080, 1920, 3) uint8 ndarray. Each call returns
    different pixel values (RNG state advances), so per-iter and per-B
    inputs vary -- no disk I/O, no cached template."""
    return _FRAME_RNG.integers(0, 256, size=_FRAME_HW, dtype=np.uint8)


def load_image_ndarray(path: Path) -> np.ndarray:
    """Backwards-compat shim. The image-from-disk path argument is ignored;
    we return a fresh random ndarray instead so the bench measures encoder
    + service work without the disk-I/O bottleneck."""
    return gen_random_frame()


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

    # --- Pre-generated input pool ---------------------------------------
    # Total B-batches consumed by FT_PE in this bench run:
    #   (TEMPORAL_SIZE + T + 2)      buffer_warmup ticks of _detect
    #   warmup_iters                 cache/JIT warmup _detect calls
    #   measure_iters                measure loop (full/half/inference/cos_sim
    #                                all share one fresh_batches() per iter)
    # Pool sized to cover all of these as unique (1080, 1920, 3) uint8
    # ndarrays so every B-batch consumer in the bench gets disjoint data.
    buffer_warmup = TEMPORAL_SIZE + frames + 2
    n_pool_batches = buffer_warmup + warmup_iters + measure_iters
    _pool_rng = np.random.default_rng()
    pool: list[np.ndarray] = [
        _pool_rng.integers(0, 256, size=_FRAME_HW, dtype=np.uint8)
        for _ in range(n_pool_batches * batch_size)
    ]
    pool_idx = [0]
    print(f"[pool] generated {len(pool)} frames "
          f"= {n_pool_batches} batches × {batch_size}  "
          f"({len(pool) * np.prod(_FRAME_HW) / 1e9:.2f} GB)  "
          f"(includes buffer_warmup={buffer_warmup} for T={frames})")

    def fresh_batches() -> list[np.ndarray]:
        """Take the next B ndarrays from the pre-generated pool. Each call
        advances the index so consecutive calls see disjoint slices."""
        i = pool_idx[0]
        if i + batch_size > len(pool):
            print(f"[pool] WARN: exhausted at {len(pool)} frames; wrapping",
                  file=sys.stderr)
            i = 0
        batch = [arr.copy() for arr in pool[i:i + batch_size]]
        pool_idx[0] = i + batch_size
        return batch

    # --- Mirror of FTPEService._postprocess_stage up to (B, 1024) video_emb,
    # omitting the text-side work (L2 norm + per-class cos-sim + alarm event
    # manager). Reuses ``service.gather_frame_buffers`` and
    # ``service.frame_buffers``, so each call advances the per-stream state
    # by exactly one tick -- the same advance ``_postprocess_stage`` causes.
    # Returns (vis_vectors, ready_stream_ids) so the text-side helper can
    # pick up where this leaves off.
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

        ready_stream_ids = []
        video_embeddings = []
        for stream_id in stream_ids:
            buf = service.frame_buffers[stream_id]
            if len(buf) >= TEMPORAL_SIZE:
                stacked = torch.stack(list(buf))                          # (TEMPORAL_SIZE, 1024)
                video_embeddings.append(stacked.mean(dim=0))              # (1024,)
                ready_stream_ids.append(stream_id)

        if not video_embeddings:
            return None, []
        return torch.stack(video_embeddings), ready_stream_ids            # (B_ready, 1024)

    # --- Mirror of the text-side tail of ``FTPEService._postprocess_stage``
    # starting from L2-norming ``vis_vectors``: per-class cos-sim against
    # text features + ``>`` + ``.cpu().tolist()`` sync + alarm event manager
    # update. Matches production cost exactly (the bench leaves
    # ``service.debug = False``, so the debug branch is skipped there too).
    def _postprocess_text_side(vis_vectors, ready_stream_ids, batches):
        user_param_map = {sid: param for sid, param in zip(stream_ids, user_params)}
        ready_set = set(ready_stream_ids)
        latest_frames = {sid: f for sid, f in zip(stream_ids, batches) if sid in ready_set}

        vis_vectors = vis_vectors / vis_vectors.norm(dim=-1, keepdim=True)

        category_preds: dict[str, list[bool]] = {}
        for class_name in ABNORMAL_CLASS_NAMES:
            txt_vec = service.category_txt_vectors.get(class_name)
            normal_vec = service.category_normal_vectors.get(class_name)
            if txt_vec is None or normal_vec is None:
                continue
            sim_abn_max = (vis_vectors @ txt_vec).max(dim=1).values
            sim_nrm_max = (vis_vectors @ normal_vec).max(dim=1).values
            category_preds[class_name] = (sim_abn_max > sim_nrm_max).detach().cpu().tolist()

        predicts_per_stream = [
            {cls: category_preds[cls][i] for cls in category_preds}
            for i in range(len(ready_stream_ids))
        ]
        user_param_list = [user_param_map[sid] for sid in ready_stream_ids]
        _ = [latest_frames[sid] for sid in ready_stream_ids]  # mirrors prod build

        return service.alarm_event_manager.update(
            predicts_per_stream, ready_stream_ids, user_param_list
        )

    # --- cos-sim dot-product helper. Times JUST the per-class cos-sim
    # matmul (vis @ cat_txt[c] and vis @ cat_normal[c]) followed by .max(1)
    # -- no L2 norm of vis_vectors, no `>` comparison, no .cpu().tolist()
    # sync, no alarm_event_manager.update. Takes a pre-stacked (B, 1024)
    # video_embeddings tensor so the cost of stacking + L2-norming
    # video_embs is excluded too.
    def _cos_sim_dot(vis_vectors):
        last = None
        for class_name in ABNORMAL_CLASS_NAMES:
            txt_vec = service.category_txt_vectors.get(class_name)
            normal_vec = service.category_normal_vectors.get(class_name)
            if txt_vec is None or normal_vec is None:
                continue
            sim_abn = (vis_vectors @ txt_vec).max(dim=1).values
            sim_nrm = (vis_vectors @ normal_vec).max(dim=1).values
            last = (sim_abn, sim_nrm)
        return last

    # --- Prime the FTPEService temporal buffer so each measured tick (full
    # OR half) actually produces a video embedding. The pool sizing above
    # already reserved ``buffer_warmup`` batches for this loop.
    for _ in range(buffer_warmup):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )

    # Additional warmup for cache/JIT. ``_detect`` covers the full vision +
    # text-side path (preprocess + buffer + model + frame_buffer + mean +
    # L2 + cos-sim + alarm), so it warms every stage the measure loop hits
    # -- including the inference-only stage, since _detect itself calls
    # _inference_stage internally on a same-shaped (B, T, C, H, W) tensor.
    for _ in range(warmup_iters):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )
    torch.cuda.synchronize()

    samples = {
        "full_cycle": [],
        "half_cycle": [],
        "inference": [],
        "input_gen_and_load": [],
        "cos_sim": [],
    }
    per_iter_temp: list[float | None] = []
    t_start = query_gpu_temp_c()

    for _ in range(measure_iters):
        # Shared full_cycle + half_cycle: one production tick of
        # preprocess + gather/frame buffer + model + per-stream mean,
        # snapshot at the half boundary, continue into the text-side
        # block (L2 norm + per-class cos-sim + alarm event manager),
        # snapshot again for full. Guarantees full_cycle >= half_cycle
        # and isolates the text-side cost (full - half) from the
        # vision-side jitter. State advances once per measure iter.
        cuda_sync()
        t0 = time.perf_counter()
        batches = fresh_batches()
        x = service._preprocess_stage(batches, user_params)
        vis_vectors, ready_stream_ids = _postprocess_to_video_emb(x, stream_ids)
        cuda_sync()
        t_half = time.perf_counter()
        if vis_vectors is not None:
            _ = _postprocess_text_side(vis_vectors, ready_stream_ids, batches)
        cuda_sync()
        t_full = time.perf_counter()
        samples["full_cycle"].append((t_full - t0) * 1000.0)
        samples["half_cycle"].append((t_half - t0) * 1000.0)

        # inference: re-time _inference_stage on a (B, T, C, H, W) tensor
        # built from THIS iter's preprocessed ``x`` (shape (B, C, H, W)),
        # so the inference measurement uses the same fresh_batches() input
        # as the shared full/half run. Construction (.contiguous()) happens
        # outside time_call so its copy doesn't bleed into the timing.
        preprocessed_bt_iter = (
            x.unsqueeze(1).expand(-1, frames, -1, -1, -1).contiguous()
        )                                                                   # (B, T, C, H, W)
        _, dt = time_call(lambda: service._inference_stage(preprocessed_bt_iter))
        samples["inference"].append(dt * 1000.0)

        # input_gen_and_load: isolated cost of producing B (1080, 1920, 3)
        # uint8 ndarrays for one tick's input (random gen here, equivalent
        # to "load from disk / camera buffer" at the start of a real tick).
        _, dt = time_call(
            lambda: [gen_random_frame() for _ in range(batch_size)]
        )
        samples["input_gen_and_load"].append(dt * 1000.0)

        # cos_sim: isolated cost of the per-class cos-sim matmul. JUST the
        # dot product (and the .max(1) reduction that immediately follows
        # inside _postprocess_stage) -- no L2 norm of video_embs, no `>`
        # comparison, no alarm event manager. Uses THIS iter's vis_vectors
        # produced by the shared full+half run, so cos_sim is on the same
        # fresh_batches() input as the rest.
        _, dt = time_call(lambda: _cos_sim_dot(vis_vectors))
        samples["cos_sim"].append(dt * 1000.0)

        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    stage_stats = {k: stats(v) for k, v in samples.items()}
    # Unit: frames encoded per second (B*T per tick / per call).
    #   - full / half_cycle: stride=1 with window_size=T means each tick
    #     re-encodes a (B, T) window = B*T frames per tick.
    #   - inference: one call encodes a (B, T) tensor, i.e. B*T frames.
    # All three are directly comparable. Per-tick *input* rate (production
    # streams per second) is reported separately for the tick-based stages.
    total_frames = batch_size * frames
    throughput = {
        f"{k}_imgs_per_s": round(total_frames * 1000.0 / stage_stats[k]["mean_ms"], 2)
        for k in samples
    }
    for k in ("full_cycle", "half_cycle"):
        throughput[f"{k}_streams_per_s"] = round(
            batch_size * 1000.0 / stage_stats[k]["mean_ms"], 2
        )

    iterations = {
        "iter": list(range(measure_iters)),
        "full_cycle_ms":          [round(v, 3) for v in samples["full_cycle"]],
        "half_cycle_ms":          [round(v, 3) for v in samples["half_cycle"]],
        "inference_ms":           [round(v, 3) for v in samples["inference"]],
        "input_gen_and_load_ms":  [round(v, 3) for v in samples["input_gen_and_load"]],
        "cos_sim_ms":             [round(v, 3) for v in samples["cos_sim"]],
        "gpu_temp_c":             [round(t, 1) if t is not None else None for t in per_iter_temp],
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
