"""Build a TRT engine with a dynamic-batch optimization profile.

Same TRT 10 STRONGLY_TYPED / BF16 / FP16 build modes as
claude_exp3_phase2/build_engine_py.py, but extends with `--min_shape`,
`--opt_shape`, `--max_shape` so a single engine runs at any batch in
[min, max].
"""
from __future__ import annotations
import argparse
import os
import time
from typing import Tuple

import tensorrt as trt

_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_shape(s: str) -> Tuple[str, Tuple[int, ...]]:
    name, _, dims = s.partition(":")
    return name, tuple(int(x) for x in dims.split("x"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--save_engine", required=True)
    p.add_argument("--min_shape", required=True, help="e.g. input:3x3x336x336")
    p.add_argument("--opt_shape", required=True)
    p.add_argument("--max_shape", required=True)
    p.add_argument("--mode", choices=["fp16", "bf16", "strongly_typed"], default="bf16")
    p.add_argument("--workspace_GiB", type=float, default=8.0)
    return p.parse_args()


def main():
    args = parse_args()
    name, min_dims = parse_shape(args.min_shape)
    _, opt_dims = parse_shape(args.opt_shape)
    _, max_dims = parse_shape(args.max_shape)
    assert len(min_dims) == len(opt_dims) == len(max_dims)

    print(f"[build_dyn] mode={args.mode} input={name}  "
          f"min={min_dims} opt={opt_dims} max={max_dims}")

    if args.mode == "strongly_typed":
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        flags = 0

    builder = trt.Builder(_LOGGER)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, _LOGGER)
    if not parser.parse_from_file(args.onnx):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")
    print(f"[build_dyn] parsed ONNX: {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(args.workspace_GiB * (1 << 30)))
    if args.mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.mode == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    profile = builder.create_optimization_profile()
    profile.set_shape(name, min=min_dims, opt=opt_dims, max=max_dims)
    config.add_optimization_profile(profile)

    print(f"[build_dyn] building engine (dynamic profile)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.save_engine)), exist_ok=True)
    with open(args.save_engine, "wb") as f:
        f.write(serialized)
    sz = os.path.getsize(args.save_engine) / 1024**2
    print(f"[build_dyn] DONE in {elapsed:.1f}s  size={sz:.1f} MiB  -> {args.save_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
