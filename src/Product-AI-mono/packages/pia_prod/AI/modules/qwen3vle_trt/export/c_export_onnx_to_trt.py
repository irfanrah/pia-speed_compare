"""
One-time script: convert ONNX models → TensorRT .engine files.

Reads Vision.onnx + Transformer.onnx from a directory and writes
Vision.engine + Transformer.engine to the same directory.

Mixed precision: FP16 globally, with normalization-related layers forced to
FP32 to avoid overflow / NaN in LayerNorm and manual RMSNorm ops.
Pass --fp32 to force full FP32 (slower, ~2× memory).

Usage:
    python -m pia_prod.AI.modules.qwen3vle_trt.export.c_export_onnx_to_trt
    python -m pia_prod.AI.modules.qwen3vle_trt.export.c_export_onnx_to_trt --fp32
"""

import argparse
import os

import numpy as np
import tensorrt as trt

from pia_prod.AI.modules.qwen3vle_trt.config import QWEN3VLE_TRT_ONNX_DIR_PATH

DEFAULT_SEQ_LEN_MIN = 16
DEFAULT_SEQ_LEN_OPT = 1024
DEFAULT_SEQ_LEN_MAX = 8192
WORKSPACE_GB = 8


def _force_norm_layers_fp32(network, verbose: bool = False) -> int:
    """
    Safe mixed-precision policy: keep compute-heavy layers (MatMul, Conv) in FP16
    and force numerically sensitive layers (norms, reductions, elementwise) to FP32.

    Why this policy:
      - In Qwen3-VL, RMSNorm is manually implemented as `x * rsqrt(square(x).sum())`.
        `x.square()` exports as ELEMENTWISE Mul(x, x), not Pow — summing hidden_size
        squared values easily overflows FP16 (max ≈ 65504).
      - Softmax (exp, sub, div) and other reductions have similar overflow risks.
      - MatMul / Conv are numerically robust in FP16 (tensor cores) and are where
        most memory and compute live, so the memory benefit is preserved.

    Forced to FP32:
      • NORMALIZATION  — fused LayerNorm
      • REDUCE         — ReduceSum / ReduceMean
      • UNARY          — Sqrt, Recip, Exp, Log, Neg, …
      • ELEMENTWISE    — Mul (squaring / final norm step), Pow, Div, …
    """
    fp32_types = {
        trt.LayerType.NORMALIZATION,
        trt.LayerType.REDUCE,
        trt.LayerType.UNARY,
        trt.LayerType.ELEMENTWISE,
    }

    count = 0
    forced_names = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in fp32_types:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.FLOAT)
            count += 1
            if verbose:
                forced_names.append(f"    {layer.type.name:<14} {layer.name}")

    if verbose and forced_names:
        print("\n".join(forced_names))
    return count


def _build_one(onnx_path: str, engine_path: str, workspace_gb: int, profile_shapes=None, fp16: bool = True, verbose_fp32: bool = False):
    """Build a single TensorRT engine from an ONNX file."""
    if os.path.exists(engine_path):
        print(f"[skip] {engine_path} already exists")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read(), path=onnx_path):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            raise RuntimeError(f"ONNX parsing failed for {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        # Respect the per-layer FP32 overrides we set below
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        n_forced = _force_norm_layers_fp32(network, verbose=verbose_fp32)
        print(f"  FP16 enabled; forced {n_forced} normalization-related layers to FP32")

    if profile_shapes:
        profile = builder.create_optimization_profile()
        for name, (min_s, opt_s, max_s) in profile_shapes.items():
            profile.set_shape(name, min_s, opt_s, max_s)
        config.add_optimization_profile(profile)

    print(f"[build] {onnx_path} -> {engine_path}")
    print("  Building engine ... (this takes a few minutes for large models)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[done]  {engine_path}  ({os.path.getsize(engine_path) / (1024 ** 3):.2f} GB)")


def export_to_trt(
    onnx_dir: str,
    workspace_gb: int = WORKSPACE_GB,
    seq_len_min: int = DEFAULT_SEQ_LEN_MIN,
    seq_len_opt: int = DEFAULT_SEQ_LEN_OPT,
    seq_len_max: int = DEFAULT_SEQ_LEN_MAX,
    fp16: bool = True,
    verbose_fp32: bool = False,
):
    """Build Vision.engine + Transformer.engine from ONNX files in onnx_dir."""
    rp = np.load(os.path.join(onnx_dir, "rotary_params.npz"))
    hidden_size = int(rp["hidden_size"])
    head_dim = int(rp["head_dim"])

    # Vision — fully static shape, no optimisation profile needed
    _build_one(
        os.path.join(onnx_dir, "Vision.onnx"),
        os.path.join(onnx_dir, "Vision.engine"),
        workspace_gb=workspace_gb,
        fp16=fp16,
        verbose_fp32=verbose_fp32,
    )

    # Transformer — dynamic seq_len, needs optimisation profile
    transformer_onnx = os.path.join(onnx_dir, "Transformer.onnx")

    import onnx
    model = onnx.load(transformer_onnx, load_external_data=False)
    input_names = [i.name for i in model.graph.input]
    deepstack_names = [n for n in input_names if n.startswith("deepstack_features_")]
    del model

    profile_shapes = {
        "hidden_states": (
            (1, seq_len_min, hidden_size),
            (1, seq_len_opt, hidden_size),
            (1, seq_len_max, hidden_size),
        ),
        "rotary_cos": (
            (1, seq_len_min, 1, 1, head_dim),
            (1, seq_len_opt, 1, 1, head_dim),
            (1, seq_len_max, 1, 1, head_dim),
        ),
        "rotary_sin": (
            (1, seq_len_min, 1, 1, head_dim),
            (1, seq_len_opt, 1, 1, head_dim),
            (1, seq_len_max, 1, 1, head_dim),
        ),
        "attention_mask": (
            (1, 1, 1, seq_len_min, seq_len_min),
            (1, 1, 1, seq_len_opt, seq_len_opt),
            (1, 1, 1, seq_len_max, seq_len_max),
        ),
    }
    for name in deepstack_names:
        profile_shapes[name] = (
            (1, seq_len_min, hidden_size),
            (1, seq_len_opt, hidden_size),
            (1, seq_len_max, hidden_size),
        )

    _build_one(
        transformer_onnx,
        os.path.join(onnx_dir, "Transformer.engine"),
        workspace_gb=workspace_gb,
        profile_shapes=profile_shapes,
        fp16=fp16,
        verbose_fp32=verbose_fp32,
    )

    print(f"\nDone. Engines written to: {onnx_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TRT engines from ONNX exports")
    parser.add_argument("--onnx-dir", default=QWEN3VLE_TRT_ONNX_DIR_PATH)
    parser.add_argument("--workspace-gb", type=int, default=WORKSPACE_GB)
    parser.add_argument("--seq-len-min", type=int, default=DEFAULT_SEQ_LEN_MIN)
    parser.add_argument("--seq-len-opt", type=int, default=DEFAULT_SEQ_LEN_OPT)
    parser.add_argument("--seq-len-max", type=int, default=DEFAULT_SEQ_LEN_MAX)
    parser.add_argument(
        "--fp32", action="store_true",
        help="Force full FP32 (disables default FP16 mixed precision)",
    )
    parser.add_argument(
        "--verbose-fp32", action="store_true",
        help="Print the names of layers forced to FP32 (debugging)",
    )
    args = parser.parse_args()

    export_to_trt(
        onnx_dir=args.onnx_dir,
        workspace_gb=args.workspace_gb,
        seq_len_min=args.seq_len_min,
        seq_len_opt=args.seq_len_opt,
        seq_len_max=args.seq_len_max,
        fp16=not args.fp32,
        verbose_fp32=args.verbose_fp32,
    )
