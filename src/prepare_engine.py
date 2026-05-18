"""Download an ONNX model from HuggingFace (if missing) and export to TRT.

Reuses the export functions inside
``pia_prod.AI.modules.{perception_encoder,ft_pe}.trt_export`` so engines built
here are byte-compatible with what the production services consume.

Usage:
    python3 src/prepare_engine.py --kind pe \
        --hf-repo PIA-SPACE-LAB/PE-Core-L14-336 \
        --hf-file onnx/PE-Core-L14-336_vision_dynamic.onnx \
        --onnx  assets/model/PE-Core-L14-336_vision_dynamic.onnx \
        --engine assets/model/PE-Core-L14-336_vision_dynamic.engine

    python3 src/prepare_engine.py --kind ftpe \
        --hf-repo <repo> --hf-file <file.onnx> \
        --onnx <local.onnx> --engine <local.engine>
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "Product-AI-mono" / "packages"))


def download_onnx(hf_repo: str, hf_file: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[prepare] ONNX already at {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] downloading {hf_file} from {hf_repo} ...")
    cached = hf_hub_download(repo_id=hf_repo, filename=hf_file)
    if Path(cached).resolve() != dest.resolve():
        shutil.copy(cached, dest)
    print(f"[prepare] ONNX ready at {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def export_pe(onnx: Path, engine: Path, min_b: int, opt_b: int, max_b: int) -> None:
    from pia_prod.AI.modules.perception_encoder.trt_export import export_trt_engine

    export_trt_engine(
        onnx_file=str(onnx),
        save_file_path=str(engine),
        input_size=(3, 336, 336),
        min_batch_size=min_b,
        opt_batch_size=opt_b,
        max_batch_size=max_b,
        half_precision=True,
    )


def export_ftpe(
    onnx: Path,
    engine: Path,
    min_b: int,
    opt_b: int,
    max_b: int,
    min_t: int,
    opt_t: int,
    max_t: int,
) -> None:
    from pia_prod.AI.modules.ft_pe.trt_export import export_trt_engine

    export_trt_engine(
        onnx_file=str(onnx),
        save_file_path=str(engine),
        min_batch_size=min_b,
        opt_batch_size=opt_b,
        max_batch_size=max_b,
        min_frames=min_t,
        opt_frames=opt_t,
        max_frames=max_t,
        height=336,
        width=336,
        half_precision=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ONNX + export TRT engine")
    p.add_argument("--kind", choices=["pe", "ftpe"], required=True)
    p.add_argument("--hf-repo", required=True)
    p.add_argument("--hf-file", required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--min-frames", type=int, default=1, help="ftpe only")
    p.add_argument("--opt-frames", type=int, default=8, help="ftpe only")
    p.add_argument("--max-frames", type=int, default=16, help="ftpe only")
    p.add_argument("--extra-hf-file", action="append", default=[],
                   help="Additional repo-relative file to fetch (e.g. text_features.json). "
                        "Pass as REPO_PATH:LOCAL_PATH or just REPO_PATH (saved next to --onnx).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Extra companion files (e.g. text_features.json) are always fetched if
    # missing, even when the engine already exists.
    for spec in args.extra_hf_file:
        if ":" in spec:
            repo_path, local_path = spec.split(":", 1)
            dest = Path(local_path)
        else:
            repo_path = spec
            dest = args.onnx.parent / Path(repo_path).name
        download_onnx(args.hf_repo, repo_path, dest)

    if args.engine.exists():
        print(f"[prepare] engine already at {args.engine} "
              f"({args.engine.stat().st_size / 1e6:.1f} MB) - skipping")
        return 0

    download_onnx(args.hf_repo, args.hf_file, args.onnx)

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "pe":
        export_pe(args.onnx, args.engine, args.min_batch, args.opt_batch, args.max_batch)
    else:
        export_ftpe(
            args.onnx, args.engine,
            args.min_batch, args.opt_batch, args.max_batch,
            args.min_frames, args.opt_frames, args.max_frames,
        )
    print(f"[prepare] engine ready at {args.engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
