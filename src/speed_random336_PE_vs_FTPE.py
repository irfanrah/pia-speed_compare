"""Inference-only latency for PE (T=1) vs FT_PE (T=3) on random
(B, 3, 336, 336) / (B, T, 3, 336, 336) tensors. One chart, one metric
(mean inference ms vs batch size). 30 s cool-down between batch sizes.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


class TRTRunner:
    """Minimal TRT engine wrapper that handles arbitrary dynamic output
    shapes — allocates output buffers lazily after the input shape is set,
    so engines with multi-dim dynamic outputs (FT_PE: -1, -1, 1024) work."""

    def __init__(self, engine_path: str) -> None:
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.tensor_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        self.inp_names = [
            n for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.out_names = [
            n for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        self.torch_input_dtypes = [
            _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)] for n in self.inp_names
        ]
        self.torch_output_dtypes = [
            _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)] for n in self.out_names
        ]
        self._out_bufs: dict[str, torch.Tensor] = {}
        self._out_shapes: dict[str, tuple[int, ...]] = {}

    @property
    def input_name(self) -> str:
        return self.inp_names[0]

    def input_shape_template(self) -> list[int]:
        return list(self.engine.get_tensor_shape(self.input_name))

    def bind(self, x: torch.Tensor) -> None:
        """One-time-per-shape setup: lock input shape, allocate output
        buffers, install pointer bindings. Call once before the timed
        inference loop so the loop body is purely the forward pass."""
        in_name = self.inp_names[0]
        self.context.set_input_shape(in_name, tuple(x.shape))
        for out_name in self.out_names:
            shape = tuple(self.context.get_tensor_shape(out_name))
            if self._out_shapes.get(out_name) != shape:
                self._out_bufs[out_name] = torch.empty(
                    shape,
                    dtype=self.torch_output_dtypes[self.out_names.index(out_name)],
                    device="cuda",
                )
                self._out_shapes[out_name] = shape
        self.context.set_tensor_address(in_name, int(x.data_ptr()))
        for out_name in self.out_names:
            self.context.set_tensor_address(
                out_name, int(self._out_bufs[out_name].data_ptr())
            )

    def forward(self) -> torch.Tensor:
        """Pure GPU forward pass — the equivalent of
        ``out = vision_encoder(processed_image)`` after ``bind(x)`` has
        installed the input. Nothing else runs in this call."""
        ok = self.context.execute_v3() if hasattr(self.context, "execute_v3") \
            else self.context.execute_async_v3(stream_handle=0)
        if not ok:
            raise RuntimeError("TensorRT inference failed.")
        return self._out_bufs[self.out_names[0]]

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: bind + forward in one call (kept for non-timed use)."""
        self.bind(x)
        return self.forward()


DEFAULT_PE_ENGINE = REPO_ROOT / "assets" / "model" / "PE-Core-L14-336.engine"
DEFAULT_FTPE_ENGINE = REPO_ROOT / "assets" / "model" / "FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine"
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]


def gpu_temp_c(device_index: int = 0) -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def profile_b_bounds(engine: trt.ICudaEngine, name: str) -> tuple[int, int]:
    mn, _opt, mx = engine.get_tensor_profile_shape(name, 0)
    return int(mn[0]), int(mx[0])


def time_inference(infer: TRTRunner, x: torch.Tensor, warmup: int, iters: int) -> list[float]:
    # One-time setup: lock shape, allocate output buffers, install bindings.
    # After this, only the GPU forward pass runs inside the timed loop —
    # equivalent to `out = vision_encoder(processed_image)`.
    infer.bind(x)
    for _ in range(warmup):
        infer.forward()
    torch.cuda.synchronize()
    out: list[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        infer.forward()                       # out = vision_encoder(x)
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def bench_model(label: str, engine_path: Path, batches: list[int], frames: int,
                size: int, warmup: int, iters: int, delay: int,
                device_index: int) -> dict:
    if not engine_path.exists():
        raise FileNotFoundError(f"{label} engine not found: {engine_path}")
    print(f"\n=== {label} | {engine_path.name} ===")
    infer = TRTRunner(str(engine_path))
    input_name = infer.input_name
    input_dtype = infer.torch_input_dtypes[0]
    min_b, max_b = profile_b_bounds(infer.engine, input_name)
    is_temporal = len(infer.input_shape_template()) == 5
    print(f"input={input_name} dtype={input_dtype} profile_B=[{min_b},{max_b}] "
          f"temporal={is_temporal}")

    rng = torch.Generator(device="cuda").manual_seed(0)
    rows: list[dict] = []
    for idx, B in enumerate(batches):
        entry: dict = {"batch": B}
        if B < min_b or B > max_b:
            print(f"  [B={B:3d}] SKIP (outside [{min_b},{max_b}])")
            entry["status"] = "SKIP_PROFILE"
            rows.append(entry)
            continue
        if idx > 0 and delay > 0:
            print(f"  [B={B:3d}] cooldown {delay}s ...")
            time.sleep(delay)
        try:
            shape = (B, frames, 3, size, size) if is_temporal else (B, 3, size, size)
            x = torch.randn(shape, dtype=input_dtype, device="cuda",
                            generator=rng).contiguous()
            samples = time_inference(infer, x, warmup, iters)
            del x
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as ex:
            print(f"  [B={B:3d}] OOM: {ex}")
            entry["status"] = "OOM"
            torch.cuda.empty_cache()
            rows.append(entry)
            continue
        except Exception as ex:
            print(f"  [B={B:3d}] ERROR: {type(ex).__name__}: {ex}")
            entry["status"] = "ERROR"
            entry["error"] = f"{type(ex).__name__}: {ex}"
            rows.append(entry)
            continue
        mean = round(statistics.fmean(samples), 3)
        std = round(statistics.pstdev(samples), 3) if len(samples) > 1 else 0.0
        entry.update({
            "status": "OK",
            "mean_ms": mean,
            "std_ms": std,
            "min_ms": round(min(samples), 3),
            "max_ms": round(max(samples), 3),
            "temp_c": gpu_temp_c(device_index),
        })
        rows.append(entry)
        print(f"  [B={B:3d}] OK  mean={mean:8.3f} ms (std={std:5.3f})  "
              f"temp={entry['temp_c']} C")
    return {
        "label": label,
        "engine": str(engine_path),
        "frames": frames if is_temporal else 1,
        "temporal": is_temporal,
        "rows": rows,
    }


def plot_compare(pe: dict, ftpe: dict, batches: list[int], out_png: Path,
                 gpu_name: str) -> None:
    def xy(res: dict) -> tuple[list[int], list[float]]:
        bs, ms = [], []
        for r in res["rows"]:
            if r.get("status") == "OK":
                bs.append(r["batch"])
                ms.append(r["mean_ms"])
        return bs, ms

    fig, ax = plt.subplots(figsize=(9, 5.5))
    pe_bs, pe_ms = xy(pe)
    ft_bs, ft_ms = xy(ftpe)
    T = ftpe["frames"]
    pe_ms_per_img = [v / b for v, b in zip(pe_ms, pe_bs)]
    ft_ms_per_img = [v / (b * T) for v, b in zip(ft_ms, ft_bs)]

    ax.plot(pe_bs, pe_ms, marker="o", color="C0", label="PE total (T=1)")
    ax.plot(ft_bs, ft_ms, marker="s", color="C1", label=f"FT_PE total (T={T})")
    ax.plot(pe_bs, pe_ms_per_img, marker="o", color="C0", linestyle=":",
            label="PE per-image (÷B)")
    ax.plot(ft_bs, ft_ms_per_img, marker="s", color="C1", linestyle=":",
            label=f"FT_PE per-image (÷(B×{T}))")

    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("inference latency (ms)")
    ax.set_yscale("log")
    ax.set_title(f"{gpu_name} — random 336 inference: PE vs FT_PE T={T}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    for x, y in zip(pe_bs, pe_ms):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="C0")
    for x, y in zip(ft_bs, ft_ms):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="C1")
    for x, y in zip(pe_bs, pe_ms_per_img):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="C0",
                    alpha=0.75)
    for x, y in zip(ft_bs, ft_ms_per_img):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="C1",
                    alpha=0.75)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"plot: {out_png}")


def parse_batches(spec: str) -> list[int]:
    return [int(t) for t in spec.split(",") if t.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pe-engine", type=Path, default=DEFAULT_PE_ENGINE)
    p.add_argument("--ftpe-engine", type=Path, default=DEFAULT_FTPE_ENGINE)
    p.add_argument("--batches", type=parse_batches, default=DEFAULT_BATCHES)
    p.add_argument("--frames", type=int, default=3, help="T for FT_PE")
    p.add_argument("--size", type=int, default=336)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--delay", type=int, default=30)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = torch.cuda.get_device_name(args.device_index)
    gpu_tag = gpu_name.replace(" ", "_")

    started = datetime.now().isoformat(timespec="seconds")
    pe_res = bench_model("PE", args.pe_engine, args.batches, frames=1,
                         size=args.size, warmup=args.warmup, iters=args.iters,
                         delay=args.delay, device_index=args.device_index)
    print(f"\n--- inter-model cooldown {args.delay}s ---")
    time.sleep(args.delay)
    ftpe_res = bench_model("FT_PE", args.ftpe_engine, args.batches,
                           frames=args.frames, size=args.size,
                           warmup=args.warmup, iters=args.iters,
                           delay=args.delay, device_index=args.device_index)
    ended = datetime.now().isoformat(timespec="seconds")

    result = {
        "gpu": gpu_name,
        "batches": args.batches,
        "frames_ftpe": args.frames,
        "warmup": args.warmup,
        "iters": args.iters,
        "delay_s": args.delay,
        "started_at": started,
        "ended_at": ended,
        "pe": pe_res,
        "ftpe": ftpe_res,
    }
    out_json = args.out_dir / f"pe_vs_ftpe_random336_{gpu_tag}_t{args.frames}_{ts}.json"
    out_png = args.out_dir / f"pe_vs_ftpe_random336_{gpu_tag}_t{args.frames}_{ts}.png"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote: {out_json}")
    plot_compare(pe_res, ftpe_res, args.batches, out_png, gpu_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
