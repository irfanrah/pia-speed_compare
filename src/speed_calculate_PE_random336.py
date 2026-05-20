"""Pure vision-encoder inference latency for PE TRT (BF16) on synthetic
(B, 3, 336, 336) inputs.

Skips PEService entirely: random float tensors are pre-generated on the
GPU in the engine's expected dtype, and only the TRT execution call is
timed. One sweep value per batch size; powers-of-two by default
(1, 2, 4, ..., 1024). 30 s cool-down between batch sizes.

The script reads the engine's optimization-profile bounds and skips any
batch size outside [min_B, max_B]. CUDA OOM at run time is caught per-B
and recorded as ``status="OOM"`` so larger B values still get a chance.
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
sys.path.insert(0, str(REPO_ROOT / "src" / "Product-AI-mono" / "packages"))

from pia_prod.AI.modules.perception_encoder.trt_load import TRTInference  # noqa: E402

DEFAULT_ENGINE = REPO_ROOT / "assets" / "model" / "PE-Core-L14-336.engine"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


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


def stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    def pct(p: float) -> float:
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        return s[k]
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "median_ms": round(statistics.median(samples_ms), 3),
        "p95_ms": round(pct(95), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "std_ms": round(statistics.pstdev(samples_ms), 3) if len(samples_ms) > 1 else 0.0,
    }


def profile_bounds(engine: trt.ICudaEngine, input_name: str) -> tuple[int, int]:
    """Return (min_B, max_B) for the engine's first optimization profile."""
    min_shape, _opt, max_shape = engine.get_tensor_profile_shape(input_name, 0)
    return int(min_shape[0]), int(max_shape[0])


def time_inference(infer: TRTInference, x: torch.Tensor, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        infer.infer(x)
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        infer.infer(x)
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return samples_ms


def run_bench(args: argparse.Namespace) -> dict:
    if not args.engine.exists():
        raise FileNotFoundError(f"engine not found: {args.engine}")

    print(f"engine: {args.engine}")
    infer = TRTInference(str(args.engine))
    input_name = infer.inp_names[0]
    input_dtype = infer.torch_input_dtypes[0]
    min_B, max_B = profile_bounds(infer.engine, input_name)
    print(f"engine profile [B]: min={min_B}  max={max_B}  dtype={input_dtype}")
    print(f"sweep B: {args.batches}  warmup={args.warmup}  iters={args.iters}  delay={args.delay}s")
    print()

    rng_gen = torch.Generator(device="cuda").manual_seed(0)
    per_batch: list[dict] = []

    for idx, B in enumerate(args.batches):
        entry: dict = {"batch": B}
        if B < min_B or B > max_B:
            print(f"[B={B:4d}] SKIP (outside engine profile [{min_B},{max_B}])")
            entry["status"] = "SKIP_PROFILE"
            per_batch.append(entry)
            continue

        if idx > 0 and args.delay > 0:
            print(f"[B={B:4d}] cooldown {args.delay}s ...")
            time.sleep(args.delay)

        temp_before = gpu_temp_c(args.device_index)
        try:
            x = torch.randn(
                (B, 3, args.size, args.size),
                dtype=input_dtype,
                device="cuda",
                generator=rng_gen,
            ).contiguous()
            samples_ms = time_inference(infer, x, args.warmup, args.iters)
            del x
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"[B={B:4d}] OOM: {e}")
            entry["status"] = "OOM"
            torch.cuda.empty_cache()
            per_batch.append(entry)
            continue
        except Exception as e:
            print(f"[B={B:4d}] ERROR: {type(e).__name__}: {e}")
            entry["status"] = "ERROR"
            entry["error"] = f"{type(e).__name__}: {e}"
            per_batch.append(entry)
            continue

        temp_after = gpu_temp_c(args.device_index)
        st = stats(samples_ms)
        imgs_per_s = round(B * 1000.0 / st["mean_ms"], 2)
        entry.update({
            "status": "OK",
            **st,
            "imgs_per_s": imgs_per_s,
            "temp_before_c": temp_before,
            "temp_after_c": temp_after,
            "iters": args.iters,
            "warmup": args.warmup,
            "samples_ms": [round(s, 3) for s in samples_ms],
        })
        per_batch.append(entry)
        print(
            f"[B={B:4d}] OK   mean={st['mean_ms']:8.3f} ms  "
            f"p95={st['p95_ms']:8.3f} ms  thr={imgs_per_s:9.2f} img/s  "
            f"temp {temp_before}->{temp_after} C"
        )

    return {
        "engine": str(args.engine),
        "input_size": args.size,
        "engine_profile_b": [min_B, max_B],
        "gpu": gpu_info(args.device_index),
        "warmup": args.warmup,
        "iters": args.iters,
        "delay_s": args.delay,
        "batches": args.batches,
        "per_batch": per_batch,
        "started_at": args.start_iso,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
    }


def plot_results(result: dict, out_png: Path) -> None:
    ok = [e for e in result["per_batch"] if e.get("status") == "OK"]
    if not ok:
        print("no OK entries to plot")
        return
    bs = [e["batch"] for e in ok]
    ms = [e["mean_ms"] for e in ok]
    thr = [e["imgs_per_s"] for e in ok]
    p95 = [e["p95_ms"] for e in ok]

    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(12, 4.8))

    sparse = len(bs) > 16  # dense sweep -> linear axis, sparse ticks

    ax_lat.plot(bs, ms, marker="o", markersize=4, label="mean")
    ax_lat.plot(bs, p95, marker="x", markersize=4, linestyle="--", label="p95", alpha=0.7)
    if sparse:
        ax_lat.set_xscale("linear")
    else:
        ax_lat.set_xscale("log", base=2)
        ax_lat.set_xticks(bs)
        ax_lat.set_xticklabels([str(b) for b in bs])
    ax_lat.set_xlabel("batch size")
    ax_lat.set_ylabel("latency (ms)")
    ax_lat.set_title("PE TRT BF16 — inference latency")
    ax_lat.grid(True, which="both", alpha=0.3)
    ax_lat.legend()

    ax_thr.plot(bs, thr, marker="o", markersize=4, color="C2")
    if sparse:
        ax_thr.set_xscale("linear")
    else:
        ax_thr.set_xscale("log", base=2)
        ax_thr.set_xticks(bs)
        ax_thr.set_xticklabels([str(b) for b in bs])
    ax_thr.set_xlabel("batch size")
    ax_thr.set_ylabel("throughput (img/s)")
    ax_thr.set_title("PE TRT BF16 — throughput")
    ax_thr.grid(True, which="both", alpha=0.3)
    if not sparse:
        for x, y in zip(bs, thr):
            ax_thr.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8)
    else:
        i_peak = thr.index(max(thr))
        ax_thr.annotate(f"peak {thr[i_peak]:.0f} @ B={bs[i_peak]}",
                        (bs[i_peak], thr[i_peak]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)

    gpu_name = result["gpu"]["name"].replace(" ", "_")
    fig.suptitle(
        f"{result['gpu']['name']}  |  "
        f"{Path(result['engine']).name}  |  "
        f"warmup={result['warmup']} iters={result['iters']} delay={result['delay_s']}s",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140)
    print(f"plot: {out_png}")


def parse_batches(spec: str) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise argparse.ArgumentTypeError("--batches must be non-empty")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    p.add_argument("--batches", type=parse_batches,
                   default=DEFAULT_BATCHES,
                   help="comma-separated batch sizes (default: 1,2,4,...,1024)")
    p.add_argument("--size", type=int, default=336)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--delay", type=int, default=30,
                   help="seconds to sleep between batch sizes (default 30)")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    args.start_iso = datetime.now().isoformat(timespec="seconds")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = torch.cuda.get_device_name(args.device_index).replace(" ", "_")
    tag = f"_{args.tag}" if args.tag else ""
    out_json = args.out_dir / f"pe_random336_{gpu_name}{tag}_{ts}.json"
    out_png = args.out_dir / f"pe_random336_{gpu_name}{tag}_{ts}.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = run_bench(args)
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote: {out_json}")
    plot_results(result, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
