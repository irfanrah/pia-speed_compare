"""Speed benchmark for FT_PE INT8 (T=3) via the real FTPEService pipeline.

Drop-in twin of ``src/speed_calculate_FTPE.py``. Same four-stage harness
(full_cycle / three_quarters_cycle / half_cycle / inference), same
production-tick semantics, same ``service.gather_frame_buffers`` /
``service.frame_buffers`` advance per call -- the only difference is the
TRT engine plugged into ``service._inference_stage``.

Engine source
-------------
The FT-T3 INT8+CRL deploy emits an engine with a **flat** input shape
``(BT, 3, 336, 336)`` (B*T flattened together), unlike the default FT_PE
engine which expects ``(B, T, 3, 336, 336)``. The PE encoder is per-image
anyway (the temporal mean-pool happens host-side), so the deploy flattens
BT before tracing and leaves the temporal-stack/mean code in Python.

To keep ``FTPEService._inference_stage`` (which hands the model a
``(B_enc, T, ...)`` tensor) usable, we wrap the INT8 engine in an adapter
that flattens to ``(B_enc * T, ...)`` for the engine call and unflattens
the ``(B_enc * T, 1024)`` output back to ``(B_enc, T, 1024)``.

Stage convention (shared with src/speed_calculate_FTPE.py):
    * ``half_cycle`` = one production tick of the vision side, stopping
      at the per-stream video embedding ``(B, 1024)``.
    * ``full_cycle`` = same tick + the text-side block (L2 norm +
      per-class cos-sim + alarm event manager).
    * ``three_quarters_cycle`` = ``full_cycle`` minus the disk read.
    * ``inference`` = pre-cooked ``(B, T, 3, H, W)`` CUDA tensor through
      the adapter (which flattens to ``(B*T, 3, H, W)`` for the engine).

Build the engine first via ``src/FTPE_INT8/scripts/run_on_a4000.sh`` and
point this script at the resulting ``int8_*_crl.engine`` (see README).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "Product-AI-mono" / "packages"))

DEFAULT_ENGINE = REPO_ROOT / "assets" / "QAT" / "int8_dyn_crl_t3.engine"
DEFAULT_TXT = REPO_ROOT / "assets" / "model" / "FT_text_features.json"
DEFAULT_IMAGE = REPO_ROOT / "assets" / "images" / "kkpolice_1.jpg"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# ---- Env-var injection BEFORE importing FTPEService -------------------------
# ft_pe.config reads these at import time. We point the engine env-var at the
# *original* FT_PE engine so FTPEService.__init__ can load successfully; the
# bench then swaps service.model for our INT8 adapter before any timed call.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                  help="INT8 TRT engine (.engine). Default: FT-T3 INT8+CRL at B=4.")
_pre.add_argument("--text-features", type=Path, default=DEFAULT_TXT)
_pre.add_argument("--ftpe-engine", type=Path, default=None,
                  help="FT_PE BF16 engine path for FTPEService bootstrap. "
                       "Default: assets/model/FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine.")
_pre_args, _ = _pre.parse_known_args()

_BOOTSTRAP_ENGINE = (
    _pre_args.ftpe_engine
    or REPO_ROOT / "assets" / "model"
    / "FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine"
)
os.environ["MODEL_PE_VIOLENCE_DETECTION_TRT_PATH"] = str(_BOOTSTRAP_ENGINE)
os.environ["FT_PE_TEXT_FEATURES_JSON"] = str(_pre_args.text_features)
os.environ.setdefault("FT_PE_MODE", "8fps")
# -----------------------------------------------------------------------------

from queue import Queue  # noqa: E402

from pia.ai.tasks.T2VRet.models.PE.utils.complexity_check import (  # noqa: E402
    time_call,
    get_gpu_stats_nvml,
)
from pia_prod.AI.modules.ft_pe.service import FTPEService  # noqa: E402
from pia_prod.AI.modules.ft_pe.config import (  # noqa: E402
    ABNORMAL_CLASS_NAMES,
    IMG_SIZE,
    TEMPORAL_SIZE,
)


_TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def _trt_to_torch_dtype(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.BF16:  torch.bfloat16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL:  torch.bool,
    }[dt]


class _Int8EngineAdapter:
    """Wraps a flat-BT TRT engine so it can stand in for ``FTPEService.model``.

    Exposes the minimal surface ``FTPEService._inference_stage`` and the
    benchmark's pre-flight check rely on: a callable ``(B, T, ...)`` input
    -> ``(B, T, 1024)`` output, plus ``self.model`` (the deserialized
    engine) and ``self.input_names`` (a 1-element list) for the pre-flight
    profile lookup."""

    def __init__(self, engine_path: Path):
        with open(engine_path, "rb") as f, trt.Runtime(_TRT_LOGGER) as rt:
            self.model = rt.deserialize_cuda_engine(f.read())
        if self.model is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.model.create_execution_context()

        names = [self.model.get_tensor_name(i)
                 for i in range(self.model.num_io_tensors)]
        self.input_names = [n for n in names
                            if self.model.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names
                             if self.model.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert len(self.input_names) == 1, \
            f"engine has {len(self.input_names)} inputs, expected 1: {self.input_names}"

        self.input_name = self.input_names[0]
        prof = self.model.get_tensor_profile_shape(self.input_name, 0)
        self.min_bt, self.opt_bt, self.max_bt = prof[0][0], prof[1][0], prof[2][0]

        self.in_dtype = _trt_to_torch_dtype(self.model.get_tensor_dtype(self.input_name))
        self.out_dtype = {
            n: _trt_to_torch_dtype(self.model.get_tensor_dtype(n))
            for n in self.output_names
        }
        # Cached output buffers, re-allocated when the dynamic BT changes.
        self._out_buffers: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._current_bt: int | None = None

    def _ensure_output_buffers(self, bt: int):
        if self._current_bt == bt and self._out_buffers:
            return
        self.context.set_input_shape(self.input_name, (bt, 3, IMG_SIZE[0], IMG_SIZE[1]))
        self._out_buffers.clear()
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            self._out_buffers[name] = torch.empty(shape, dtype=self.out_dtype[name], device="cuda")
        self._current_bt = bt

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # FTPEService._inference_stage hands us a (B, T, 3, H, W) tensor.
        # The INT8 engine wants (B*T, 3, H, W). Flatten/unflatten here.
        if x.dim() == 5:
            B, T = x.shape[0], x.shape[1]
            x_flat = x.reshape(B * T, *x.shape[2:])
        elif x.dim() == 4:
            B, T = x.shape[0], 1
            x_flat = x
        else:
            raise ValueError(f"unexpected input shape {tuple(x.shape)}")

        if x_flat.dtype != self.in_dtype:
            x_flat = x_flat.to(self.in_dtype)
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        bt = x_flat.shape[0]
        if not (self.min_bt <= bt <= self.max_bt):
            raise ValueError(
                f"BT={bt} outside engine profile [{self.min_bt}, {self.max_bt}]"
            )
        self._ensure_output_buffers(bt)

        self.context.set_tensor_address(self.input_name, int(x_flat.data_ptr()))
        for name, buf in self._out_buffers.items():
            self.context.set_tensor_address(name, int(buf.data_ptr()))

        ok = self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream,
        )
        if not ok:
            raise RuntimeError("INT8 engine execute_async_v3 returned False")
        # Sync inside time_call's bracket via cuda_sync; an explicit sync here
        # would force a per-call host sync but keep things simple.
        out = self._out_buffers[self.output_names[0]]

        # Unflatten back to (B, T, 1024). FTPEService consumes the output as
        # ``embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)``
        # then per-stream emb[-prediction_size:].unbind(0) -- both work in
        # FP32, so we float() before reshape.
        emb = out.float().reshape(B, T, -1)
        return emb


# ---- standard helpers from speed_calculate_FTPE.py --------------------------

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
    engine_path: Path,
    image_path: Path,
    batch_size: int,
    frames: int,
    warmup_iters: int,
    measure_iters: int,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not engine_path.exists():
        raise FileNotFoundError(
            f"INT8 engine not found at {engine_path}.\n"
            "Build it via src/FTPE_INT8/scripts/run_on_a4000.sh (see "
            "src/FTPE_INT8/README.md)."
        )

    service = FTPEService(Queue())

    # Swap the FT_PE bootstrap model for the INT8 adapter.
    adapter = _Int8EngineAdapter(engine_path)
    service.model = adapter

    # Pre-flight: the INT8 engine's profile is on the **flat** BT axis,
    # not (B, T). Check directly against the adapter's BT range.
    bt = batch_size * frames
    if not (adapter.min_bt <= bt <= adapter.max_bt):
        raise ValueError(
            f"requested B*T = {bt} (B={batch_size}, T={frames}) is outside the "
            f"INT8 engine's profile [{adapter.min_bt}, {adapter.max_bt}]. "
            f"Build the engine for a wider profile, or change --batch / --frames."
        )

    # Override sliding-window state so --frames T drives the full_cycle's
    # encode shape. stride=1 means every tick triggers an encode + decision.
    service.window_size = frames
    service.sliding_window_size = max(0, frames - 1)
    service.prediction_size = 1
    service.stride = max(1, service.window_size - service.sliding_window_size)

    stream_ids = [f"stream_{i}" for i in range(batch_size)]
    user_params = make_user_params(batch_size)
    template = load_image_ndarray(image_path)

    def fresh_batches() -> list[np.ndarray]:
        return [template.copy() for _ in range(batch_size)]

    # Build a (B, T, 3, H, W) tensor for the inference-only stage by
    # expanding one tick's preprocessed batch across the temporal axis.
    _single_tick = service._preprocess_stage(fresh_batches(), user_params)
    preprocessed_bt = (
        _single_tick.unsqueeze(1).expand(-1, frames, -1, -1, -1).contiguous()
    )

    # Same _postprocess_to_video_emb helper used by speed_calculate_FTPE.py:
    # mirrors _postprocess_stage up to the per-stream mean-pool (video_emb).
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
                encode_tensors.append(torch.stack(list(gbuf)))
                service._unconsumed_frames[stream_id] -= service.stride
                encode_stream_ids.append(stream_id)

        if encode_stream_ids:
            encode_input = torch.stack(encode_tensors)
            embeddings = service._inference_stage(encode_input)
            for sid, emb in zip(encode_stream_ids, embeddings):
                new_embs = emb[-service.prediction_size:]
                service.frame_buffers[sid].extend(new_embs.unbind(0))
                if len(service.frame_buffers[sid]) > TEMPORAL_SIZE:
                    del service.frame_buffers[sid][1]

        video_embeddings = []
        for stream_id in stream_ids:
            buf = service.frame_buffers[stream_id]
            if len(buf) >= TEMPORAL_SIZE:
                stacked = torch.stack(list(buf))
                video_embeddings.append(stacked.mean(dim=0))

        if not video_embeddings:
            return None
        return torch.stack(video_embeddings)

    # Text-side helper for the isolated cos_sim measurement (mirrors
    # FTPEService._postprocess_stage from L2 norm through alarm.update).
    def _text_side(video_embeddings_BxD, stream_ids, user_params):
        vis_vectors = video_embeddings_BxD / video_embeddings_BxD.norm(
            dim=-1, keepdim=True,
        )
        category_preds: dict[str, list[bool]] = {}
        for class_name in ABNORMAL_CLASS_NAMES:
            txt_vec = service.category_txt_vectors.get(class_name)
            normal_vec = service.category_normal_vectors.get(class_name)
            if txt_vec is None or normal_vec is None:
                continue
            sim_abn_max = (vis_vectors @ txt_vec).max(dim=1).values
            sim_nrm_max = (vis_vectors @ normal_vec).max(dim=1).values
            category_preds[class_name] = (
                sim_abn_max > sim_nrm_max
            ).detach().cpu().tolist()
        predicts_per_stream = [
            {cls: category_preds[cls][i] for cls in category_preds}
            for i in range(len(stream_ids))
        ]
        return service.alarm_event_manager.update(
            predicts_per_stream, stream_ids, user_params,
        )

    buffer_warmup = TEMPORAL_SIZE + service.window_size + 2
    for _ in range(buffer_warmup):
        service._detect(
            batches=fresh_batches(), stream_ids=stream_ids, user_params=user_params,
        )

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
        "disk_read": [],
        "cos_sim": [],
    }
    per_iter_temp: list[float | None] = []
    t_start = query_gpu_temp_c()

    in_mem = fresh_batches()

    # Capture a (B, 1024) video_embeddings tensor for the cos_sim stage.
    # The text-side block doesn't care about the values -- the matmul cost
    # is shape-only -- so one tick's post-warmup video emb is a fine input.
    _half_warm = _postprocess_to_video_emb(
        service._preprocess_stage(fresh_batches(), user_params), stream_ids,
    )
    if _half_warm is None:
        raise RuntimeError("warmup didn't fill the frame_buffer; cannot prime cos_sim input")
    fixed_video_embs = _half_warm.detach()

    for _ in range(measure_iters):
        def _full():
            batches = [load_image_ndarray(image_path) for _ in range(batch_size)]
            return service._detect(
                batches=batches, stream_ids=stream_ids, user_params=user_params,
            )
        _, dt = time_call(_full)
        samples["full_cycle"].append(dt * 1000.0)

        def _three_quarters():
            batches = [b.copy() for b in in_mem]
            return service._detect(
                batches=batches, stream_ids=stream_ids, user_params=user_params,
            )
        _, dt = time_call(_three_quarters)
        samples["three_quarters_cycle"].append(dt * 1000.0)

        def _half():
            batches = [load_image_ndarray(image_path) for _ in range(batch_size)]
            x = service._preprocess_stage(batches, user_params)
            return _postprocess_to_video_emb(x, stream_ids)
        _, dt = time_call(_half)
        samples["half_cycle"].append(dt * 1000.0)

        _, dt = time_call(lambda: service._inference_stage(preprocessed_bt))
        samples["inference"].append(dt * 1000.0)

        # disk_read: isolated cost of loading B ndarrays from disk
        # (PIL.Image.open + decode + np.array). ≈ full_cycle − three_quarters_cycle.
        _, dt = time_call(
            lambda: [load_image_ndarray(image_path) for _ in range(batch_size)]
        )
        samples["disk_read"].append(dt * 1000.0)

        # cos_sim: isolated text-side block (L2 norm + 4× per-class cos-sim
        # vs text features + comparison + alarm event manager). ≈ full_cycle
        # − half_cycle.
        _, dt = time_call(
            lambda: _text_side(fixed_video_embs, stream_ids, user_params)
        )
        samples["cos_sim"].append(dt * 1000.0)

        per_iter_temp.append(query_gpu_temp_c())

    t_end = query_gpu_temp_c()
    temps = [t for t in per_iter_temp if t is not None]

    stage_stats = {k: stats(v) for k, v in samples.items()}
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
        "disk_read_ms":            [round(v, 3) for v in samples["disk_read"]],
        "cos_sim_ms":              [round(v, 3) for v in samples["cos_sim"]],
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
        "engine_profile_bt": {
            "min": adapter.min_bt,
            "opt": adapter.opt_bt,
            "max": adapter.max_bt,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FT_PE INT8 (T=3) TRT speed benchmark"
    )
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                   help="INT8 TRT engine path (.engine)")
    p.add_argument("--ftpe-engine", type=Path, default=None,
                   help="BF16 FT_PE engine used only to bootstrap FTPEService.")
    p.add_argument("--text-features", type=Path, default=DEFAULT_TXT)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--frames", type=int, default=3,
                   help="Temporal window T. INT8 deploy ships T=3 as canonical; "
                        "T must satisfy B*T within the engine's BT profile.")
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

    initial_time = datetime.now()
    info = gpu_info()

    print(f"model:  FT_PE_INT8 (T=3 canonical)  via FTPEService._detect (model swapped)")
    print(f"engine: {args.engine}")
    print(f"bootstrap engine: {_BOOTSTRAP_ENGINE}")
    print(f"txtfts: {args.text_features}")
    print(f"T:      {args.frames}  (window_size driven by --frames)")
    print(f"image:  {args.image}")
    print(f"gpu:    {info['name']}  (sm {info['compute_capability']}, "
          f"{info['total_memory_mib']} MiB)")
    print(f"batch:  {args.batch}  frames: {args.frames}  "
          f"warmup: {args.warmup}  iters: {args.iters}")
    print(f"start:  {initial_time.isoformat(timespec='seconds')}")

    result = benchmark(
        engine_path=args.engine,
        image_path=args.image,
        batch_size=args.batch,
        frames=args.frames,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
    )

    payload = {
        "model": "FT_PE-Core-L14-336_INT8_CRL",
        "code_path": "FTPEService._detect (split stages, INT8 engine swapped in)",
        "initial_time": initial_time.isoformat(timespec="seconds"),
        "gpu_type": info["name"],
        "gpu": info,
        "batch_size": args.batch,
        "frames": args.frames,
        "engine_path": str(args.engine),
        "bootstrap_engine_path": str(_BOOTSTRAP_ENGINE),
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
        / f"ftpe_int8_{safe_gpu}_b{args.batch}_t{args.frames}_{ts}{suffix}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    for stage, s in result["stages"].items():
        print(f"  {stage:<22}  mean={s['mean_ms']:>8.3f} ms  "
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
