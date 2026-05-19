"""Python-API fallback for trtexec when only the TensorRT pip wheel is installed.

The pip-installed `tensorrt` wheel does not ship the `trtexec` binary. This
script builds an equivalent engine via `IBuilder` so the recipe can still
run end-to-end. It supports the two modes used by build_engines.sh:

  --mode fp16          mirrors `trtexec --fp16`
  --mode strongly_typed   mirrors `trtexec --stronglyTyped` (TRT 10's
                       NetworkDefinitionCreationFlag.STRONGLY_TYPED;
                       precision is honored from the ONNX Q/DQ graph)

Notes:
- `--noDataTransfers / --useCudaGraph / --useSpinWait` are *runtime* flags in
  trtexec; they don't apply at build time. They affect benchmark timing only,
  which is handled separately in bench_trt.py.
- Median latency is measured here so `run.sh` can still grep "median" out of
  the log and produce the same output table as trtexec.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tensorrt as trt


_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--save_engine", required=True)
    p.add_argument("--shape", required=True,
                   help="Format: NAME:NxCxHxW (e.g. input:8x3x336x336).")
    p.add_argument("--mode", choices=["fp16", "bf16", "strongly_typed"], required=True)
    p.add_argument("--workspace_GiB", type=float, default=12.0)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def parse_shape(spec: str):
    name, dims = spec.split(":")
    dims = tuple(int(d) for d in dims.split("x"))
    return name, dims


def build(args):
    name, dims = parse_shape(args.shape)
    print(f"[build_py] mode={args.mode} input={name}{dims}")

    if args.mode == "strongly_typed":
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        flags = 0

    builder = trt.Builder(_LOGGER)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, _LOGGER)
    if not parser.parse_from_file(args.onnx):
        errs = "\n".join(str(parser.get_error(i))
                         for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")
    print(f"[build_py] parsed ONNX: {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_GiB * (1 << 30)),
    )
    if args.mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.mode == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    profile = builder.create_optimization_profile()
    profile.set_shape(name, min=dims, opt=dims, max=dims)
    config.add_optimization_profile(profile)

    print(f"[build_py] building engine ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.save_engine)), exist_ok=True)
    with open(args.save_engine, "wb") as f:
        f.write(serialized)
    sz = os.path.getsize(args.save_engine) / 1024**2
    print(f"[build_py] DONE in {elapsed:.1f}s  size={sz:.1f} MiB  -> {args.save_engine}")
    return args.save_engine, dims


def benchmark(engine_path: str, dims, iters: int, warmup: int) -> float:
    import torch

    with open(engine_path, "rb") as f, trt.Runtime(_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_names = [n for n in names
                 if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    _DT = {trt.DataType.FLOAT: torch.float32,
           trt.DataType.HALF:  torch.float16,
           trt.DataType.BF16:  torch.bfloat16,
           trt.DataType.INT8:  torch.int8}
    in_dtype = _DT[engine.get_tensor_dtype(in_name)]

    x = torch.randn(*dims, dtype=in_dtype, device="cuda").contiguous()
    context.set_input_shape(in_name, tuple(x.shape))
    context.set_tensor_address(in_name, int(x.data_ptr()))

    outs = {}
    for n in out_names:
        s = tuple(int(d) for d in context.get_tensor_shape(n))
        d = _DT[engine.get_tensor_dtype(n)]
        outs[n] = torch.empty(s, dtype=d, device="cuda")
        context.set_tensor_address(n, int(outs[n].data_ptr()))

    stream = torch.cuda.current_stream().cuda_stream

    def run():
        context.execute_async_v3(stream_handle=stream)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[len(ms) // 2]


def main() -> int:
    args = parse_args()
    engine_path, dims = build(args)

    median_ms = benchmark(engine_path, dims, args.iters, args.warmup)
    # Print in a format the build_engines.sh `grep median` step can pick up.
    print(f"[build_py] median latency: {median_ms:.3f} ms over {args.iters} iters "
          f"(warmup {args.warmup})")
    print(f"GPU Compute Time: median = {median_ms:.3f} ms")

    sidecar = engine_path + ".json"
    with open(sidecar, "w") as f:
        json.dump({
            "engine_path": engine_path,
            "mode": args.mode,
            "shape": list(dims),
            "median_ms": median_ms,
            "iters": args.iters,
            "warmup": args.warmup,
        }, f, indent=2)
    print(f"[build_py] sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
