"""Insert Q/DQ pairs via nvidia-modelopt — same recipe as claude_exp2 but with
a video-frame calibration tensor.

Calibration data is built by `video_utils.build_calibration_npy(...)` from
the 5 train + 5 val videos sampled in `calib/manifest.json`. We tile to
`--target_calib_n` so n_itr = target_calib_n // BT >= 1 for every batch in
the sweep (BT up to 96 at B=32, T=3).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np  # noqa: F401  (kept for parity with claude_exp2 / debug)

sys.path.insert(0, os.path.dirname(__file__))
from video_utils import build_calibration_npy, resolve_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--img_size", type=int, default=336)
    p.add_argument("--frames_per_video", type=int, default=3)
    p.add_argument("--n_per_split", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--dataset_root",
                   default="/home/piawsa6000/nas192/Research_materials/Kur/"
                           "PIA_clip_dataset/train_val_master_v2")
    p.add_argument("--manifest",
                   default=os.path.join(os.path.dirname(__file__),
                                        "calib", "manifest.json"))
    p.add_argument("--calib_npy",
                   default=os.path.join(os.path.dirname(__file__),
                                        "calib", "calibration.npy"))
    p.add_argument("--target_calib_n", type=int, default=96,
                   help="Length of the calibration .npy. Should be >= max BT "
                        "across the sweep so modelopt can split into >=1 batches.")
    p.add_argument("--calibration_method", default="entropy",
                   choices=["entropy", "max", "mse"])
    p.add_argument("--op_types_to_exclude", default="Add")
    p.add_argument("--high_precision_dtype", default="fp16")
    p.add_argument("--quantize_mode", default="int8")
    p.add_argument("--calibration_shapes", default=None,
                   help="e.g. 'input:3x3x336x336' for B=1, T=3")
    p.add_argument("--disable_mha_qdq", action="store_true")
    p.add_argument("--calibrate_per_node", action="store_true",
                   help="Run modelopt's per-node calibration (one tensor at a "
                        "time) instead of full-graph forward. Drops the "
                        "activation peak from ~50 GB to ~1 GB at BT=96; "
                        "essential for quantizing the larger engines on "
                        "this 64 GB host.")
    p.add_argument("--stratified", action="store_true",
                   help="Stratified per-class video sampling instead of "
                        "uniform-random. Spreads the 7 PIA classes evenly.")
    p.add_argument("--exclude_patch_embed", action="store_true",
                   help="Auto-detect the first Conv node in the graph "
                        "(PE's patch-embedding stem) and pass it through "
                        "--nodes_to_exclude. Keeps it in BF16 — typical +0.002"
                        " — +0.005 cos win because raw-pixel input has the "
                        "widest dynamic range in the graph.")
    p.add_argument("--extra_exclude_nodes", default="",
                   help="Comma-separated additional regex patterns for "
                        "--nodes_to_exclude (after the patch-embed pattern).")
    p.add_argument("--simplify", action="store_true",
                   help="Run modelopt's --simplify (uses onnxsim) before "
                        "quantization. Folds the PE-trace `If` constant so "
                        "--calibrate_per_node doesn't trip topo-sort errors.")
    p.add_argument("--autotune", default="",
                   choices=["", "quick", "default", "extensive"],
                   help="Enable modelopt --autotune Q/DQ placement search.")
    p.add_argument("--split", default=None,
                   help="Restrict the calibration video pool to a specific "
                        "dataset split (e.g. 'val'). Default: legacy "
                        "train+val pool. Use 'val' for the data-hygiene "
                        "pipeline so PTQ calibration sits between QAT-train "
                        "and the test-split final benchmark.")
    p.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def find_patch_embed_conv_name(onnx_path: str) -> str:
    """Return the name of the first Conv node in the model graph.

    PE's `visual.conv1` is the patch-embedding stem: a 14×14 stride-14
    Conv that turns the raw image into 24×24 token features. It's the
    only Conv node before the transformer body, so "first Conv" is a
    reliable identifier without hardcoding torch module paths in the
    ONNX-side regex.
    """
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    for node in model.graph.node:
        if node.op_type == "Conv":
            return node.name
    raise RuntimeError("no Conv node found in graph; this is unexpected for "
                       "a PE vision tower")


def shape_infer_in_place(onnx_path: str) -> str:
    import onnx
    inferred = onnx_path.replace(".onnx", ".inferred.onnx")
    print(f"[quantize] running ONNX shape inference: {onnx_path} -> {inferred}")
    try:
        onnx.shape_inference.infer_shapes_path(onnx_path, inferred,
                                               check_type=False,
                                               strict_mode=False,
                                               data_prop=True)
    except Exception as e:
        print(f"[quantize] shape inference failed ({e}); using original ONNX")
        return onnx_path
    if not os.path.isfile(inferred):
        return onnx_path
    return inferred


def ensure_calibration_npy(args) -> str:
    if os.path.isfile(args.calib_npy):
        arr = np.load(args.calib_npy, mmap_mode="r")
        if arr.shape[0] >= args.target_calib_n and arr.shape[1:] == (
                3, args.img_size, args.img_size):
            print(f"[quantize] reusing calibration tensor: {arr.shape} -> "
                  f"{args.calib_npy}")
            return args.calib_npy
    splits_kw = {"splits": (args.split,)} if args.split else {}
    samples = resolve_samples(manifest_path=args.manifest,
                              dataset_root=args.dataset_root,
                              n_per_split=args.n_per_split,
                              seed=args.seed,
                              stratified=args.stratified,
                              **splits_kw)
    print(f"[quantize] building calibration tensor from "
          f"{len(samples)} videos × T={args.frames_per_video} -> "
          f"target N={args.target_calib_n}")
    os.makedirs(os.path.dirname(args.calib_npy), exist_ok=True)
    build_calibration_npy(samples,
                          n_frames=args.frames_per_video,
                          img_size=args.img_size,
                          target_n=args.target_calib_n,
                          out_path=args.calib_npy)
    arr = np.load(args.calib_npy, mmap_mode="r")
    print(f"[quantize] wrote calibration tensor: {arr.shape} -> {args.calib_npy}")
    return args.calib_npy


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.onnx_path):
        raise FileNotFoundError(args.onnx_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    args.onnx_path = shape_infer_in_place(args.onnx_path)
    calib_npy = ensure_calibration_npy(args)

    cmd = [
        sys.executable, "-m", "modelopt.onnx.quantization",
        f"--onnx_path={args.onnx_path}",
        f"--quantize_mode={args.quantize_mode}",
        f"--high_precision_dtype={args.high_precision_dtype}",
        f"--op_types_to_exclude={args.op_types_to_exclude}",
        f"--output_path={args.output_path}",
        f"--calibration_data_path={calib_npy}",
        f"--calibration_method={args.calibration_method}",
    ]
    if args.calibration_shapes:
        cmd.append(f"--calibration_shapes={args.calibration_shapes}")
    if args.disable_mha_qdq:
        cmd.append("--disable_mha_qdq")
    if args.calibrate_per_node:
        cmd.append("--calibrate_per_node")
    if args.simplify:
        cmd.append("--simplify")
    if args.autotune:
        cmd += ["--autotune", args.autotune]

    exclude_patterns = []
    if args.exclude_patch_embed:
        try:
            patch_node = find_patch_embed_conv_name(args.onnx_path)
            print(f"[quantize] excluding patch-embed Conv node: {patch_node!r}")
            # Match the node name verbatim, plus any Q/DQ pair feeding/leaving
            # it that the modelopt static-quant pass would otherwise insert.
            exclude_patterns.append(patch_node)
        except Exception as e:
            print(f"[quantize] WARN: could not auto-detect patch-embed Conv "
                  f"({e}); skipping --exclude_patch_embed")
    if args.extra_exclude_nodes:
        exclude_patterns += [p for p in args.extra_exclude_nodes.split(",") if p]
    if exclude_patterns:
        cmd += ["--nodes_to_exclude", *exclude_patterns]

    if args.extra_args:
        cmd += args.extra_args

    print(f"[quantize] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[quantize] modelopt CLI failed (rc={rc})", file=sys.stderr)
        return rc

    out_size = os.path.getsize(args.output_path) / 1024**2
    print(f"[quantize] DONE -> {args.output_path}  ({out_size:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
