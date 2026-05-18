"""Plot per-iteration timings + GPU temperature for each benchmark JSON.

For every ``results/*.json`` written by ``speed_calculate_PE.py`` or
``speed_calculate_FTPE.py``, write a ``<same_name>.png`` next to it.

Left Y axis  : full_cycle, three_quarters_cycle, half_cycle, inference (ms)
Right Y axis : GPU temperature (deg C)
X axis       : iteration index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def plot_one(payload: dict, out_path: Path) -> None:
    it = payload.get("iterations") or {}
    if not it or "iter" not in it:
        raise ValueError("payload missing 'iterations' block")

    x = it["iter"]
    full = it.get("full_cycle_ms", [])
    tqc  = it.get("three_quarters_cycle_ms", [])
    half = it.get("half_cycle_ms", [])
    inf  = it.get("inference_ms", [])
    temp = it.get("gpu_temp_c", [])

    fig, ax_l = plt.subplots(figsize=(10, 5))
    ax_l.plot(x, full, marker="o", markersize=3, label="full_cycle",            color="#1f77b4")
    if tqc:
        ax_l.plot(x, tqc, marker="D", markersize=3, label="three_quarters_cycle", color="#9467bd")
    ax_l.plot(x, half, marker="s", markersize=3, label="half_cycle",            color="#2ca02c")
    ax_l.plot(x, inf,  marker="^", markersize=3, label="inference",             color="#ff7f0e")
    ax_l.set_xlabel("iter")
    ax_l.set_ylabel("latency (ms)")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(loc="upper left")

    ax_r = ax_l.twinx()
    temp_x = [xi for xi, t in zip(x, temp) if t is not None]
    temp_y = [t for t in temp if t is not None]
    if temp_y:
        ax_r.plot(temp_x, temp_y, linestyle="--", marker="x", markersize=4,
                  label="gpu_temp", color="#d62728", alpha=0.7)
        ax_r.set_ylabel("GPU temperature (°C)", color="#d62728")
        ax_r.tick_params(axis="y", labelcolor="#d62728")
        ax_r.legend(loc="upper right")

    model = payload.get("model", "?")
    gpu = payload.get("gpu_type", "?")
    B = payload.get("batch_size", "?")
    T = payload.get("frames")
    iters = payload.get("measure_iters", len(x))
    started = payload.get("initial_time", "")
    suffix = f" T={T}" if T is not None else ""
    title = f"{model}  |  {gpu}  |  B={B}{suffix}  iters={iters}  |  {started}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot benchmark JSONs to PNG")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--pattern", type=str, default="*.json")
    p.add_argument("--files", nargs="*", type=Path,
                   help="Explicit JSON files to plot (overrides --results-dir/--pattern)")
    args = p.parse_args()

    files = list(args.files) if args.files else sorted(args.results_dir.glob(args.pattern))
    if not files:
        print(f"no result JSONs found in {args.results_dir}", file=sys.stderr)
        return 1

    n_ok = 0
    for f in files:
        if f.name == "summary.txt":
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        out = f.with_suffix(".png")
        try:
            plot_one(payload, out)
            print(f"wrote {out}")
            n_ok += 1
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)

    print(f"\nplotted {n_ok} / {len(files)} runs")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
