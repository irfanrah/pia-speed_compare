"""Aggregate benchmark JSONs in ``results/`` into a clean .txt table.

Reads every ``*.json`` produced by ``speed_calculate_PE.py`` and
``speed_calculate_FTPE.py``, then writes a single fixed-width ASCII table
ordered by model -> GPU -> batch -> timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT = REPO_ROOT / "results" / "summary.txt"


COLUMNS: list[tuple[str, str, int]] = [
    # (header, source-key-path, width)
    ("model",        "model",                                       22),
    ("gpu",          "gpu_type",                                    22),
    ("B",            "batch_size",                                   4),
    ("T",            "frames",                                       4),
    ("iters",        "measure_iters",                                6),
    ("full_ms",      "stages.full_cycle.mean_ms",                   10),
    ("full_std",     "stages.full_cycle.std_ms",                     9),
    ("3q_ms",        "stages.three_quarters_cycle.mean_ms",         10),
    ("3q_std",       "stages.three_quarters_cycle.std_ms",           9),
    ("half_ms",      "stages.half_cycle.mean_ms",                   10),
    ("half_std",     "stages.half_cycle.std_ms",                     9),
    ("inf_ms",       "stages.inference.mean_ms",                    10),
    ("inf_std",      "stages.inference.std_ms",                      9),
    ("full_p95",     "stages.full_cycle.p95_ms",                    10),
    ("inf_p95",      "stages.inference.p95_ms",                     10),
    ("inf_img/s",    "throughput.inference_imgs_per_s",             11),
    ("temp_c",       "gpu_temperature_c.max",                        7),
    ("started",      "initial_time",                                20),
]


def get_nested(d: dict, key: str):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fmt_cell(value, width: int) -> str:
    if value is None:
        return "-".ljust(width)
    if isinstance(value, float):
        return f"{value:.2f}".rjust(width)
    if isinstance(value, int):
        return str(value).rjust(width)
    s = str(value)
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s.ljust(width)


def build_row(payload: dict) -> list[str]:
    return [fmt_cell(get_nested(payload, key), w) for _, key, w in COLUMNS]


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate benchmark JSONs to a text table")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pattern", type=str, default="*.json")
    args = p.parse_args()

    files = sorted(args.results_dir.glob(args.pattern))
    if not files:
        print(f"no result JSONs found in {args.results_dir}", file=sys.stderr)
        return 1

    payloads = []
    for f in files:
        try:
            payloads.append((f, json.loads(f.read_text())))
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)

    payloads.sort(
        key=lambda fp: (
            fp[1].get("model", ""),
            fp[1].get("gpu_type", ""),
            fp[1].get("batch_size", 0),
            fp[1].get("initial_time", ""),
        )
    )

    header = "  ".join(h.ljust(w) for h, _, w in COLUMNS)
    sep = "-" * len(header)

    lines = [header, sep]
    for _, payload in payloads:
        lines.append("  ".join(build_row(payload)))

    lines.append(sep)
    lines.append(f"total: {len(payloads)} runs  source: {args.results_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
