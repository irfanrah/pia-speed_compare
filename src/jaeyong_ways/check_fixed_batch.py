"""PE-Core-L14-336 fixed-shape TRT engine checker.

Builds a TRT engine pinned to a single static batch size (min = opt = max),
then reruns the 3 × batch=N chained-schedule benchmark from check_chained.py.
TRT's optimizer can specialize kernels harder when it knows the input shape
is fixed, so a static-batch engine usually beats the dynamic-profile engine
at the same batch.

Default: FIXED_BATCH = 4 (which the dynamic engine already showed as the
sweet spot for 12-channel = 3 × batch=4 scheduling).
"""

import json
import os
import sys
import time
from typing import List

import pandas as pd
import tensorrt as trt
import torch

from src.check_perception_encoder import (
    ENGINE_PATH as DYN_ENGINE_PATH,
    IMG_SIZE,
    INPUT_SIZE,
    ONNX_PATH,
    REALTIME_BUDGET_MS,
    SAMPLE_IMAGE,
    TARGET_CHANNELS,
    TRTVisionEncoder,
    _percentile,
    device_info,
    ensure_onnx,
    make_text_bank,
    postprocess,
)
from src.preprocess import load_image, preprocess

FIXED_BATCH = 4
N_INFER = 3
FIXED_ENGINE_PATH = f"assets/model/PE-Core-L14-336_vision_b{FIXED_BATCH}.engine"
RESULTS_JSON = f"results/pe_check_fixed_b{FIXED_BATCH}.json"

WARMUP_ITERS = 10
MEASURE_ITERS = 30


def export_fixed_trt_engine(
    onnx_file: str,
    save_file_path: str,
    fixed_batch: int,
    input_size=INPUT_SIZE,
) -> str:
    if os.path.exists(save_file_path):
        print(f"Fixed-batch engine already at {save_file_path}")
        return save_file_path
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(onnx_file)
    os.makedirs(os.path.dirname(save_file_path) or ".", exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(fixed_batch, *input_size),
        opt=(fixed_batch, *input_size),
        max=(fixed_batch, *input_size),
    )
    config.add_optimization_profile(profile)

    print(f"Building fixed-batch={fixed_batch} engine ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build fixed-batch engine")
    with open(save_file_path, "wb") as f:
        f.write(serialized)
    print(f"Engine saved at {save_file_path} ({os.path.getsize(save_file_path) / 1e6:.1f} MB)")
    return save_file_path


@torch.inference_mode()
def measure(model: TRTVisionEncoder, text_bank, frame) -> dict:
    total_frames = N_INFER * FIXED_BATCH
    inp_full = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
    slices = [
        inp_full[i * FIXED_BATCH : (i + 1) * FIXED_BATCH].contiguous() for i in range(N_INFER)
    ]

    for _ in range(WARMUP_ITERS):
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()

    single_call_ms = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        _ = model(slices[0])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        single_call_ms.append((t1 - t0) * 1000)

    inf_only_ms = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        for s in slices:
            _ = model(s)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_only_ms.append((t1 - t0) * 1000)

    full_ms = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        inp = preprocess([frame] * total_frames, resize_size=IMG_SIZE, device="cuda")
        slices_now = [
            inp[i * FIXED_BATCH : (i + 1) * FIXED_BATCH].contiguous() for i in range(N_INFER)
        ]
        for s in slices_now:
            feats = model(s)
            _ = postprocess(feats, text_bank)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        full_ms.append((t1 - t0) * 1000)

    return {
        "fixed_batch": FIXED_BATCH,
        "n_infer": N_INFER,
        "channels_per_cycle": total_frames,
        "single_batch_ms_mean": round(sum(single_call_ms) / len(single_call_ms), 3),
        "single_batch_ms_p95": round(_percentile(single_call_ms, 95), 3),
        "inference_only_ms_mean": round(sum(inf_only_ms) / len(inf_only_ms), 3),
        "inference_only_ms_p95": round(_percentile(inf_only_ms, 95), 3),
        "full_cycle_ms_mean": round(sum(full_ms) / len(full_ms), 3),
        "full_cycle_ms_p95": round(_percentile(full_ms, 95), 3),
        "inference_only_meets_500ms": (sum(inf_only_ms) / len(inf_only_ms)) <= REALTIME_BUDGET_MS,
        "full_cycle_meets_500ms": (sum(full_ms) / len(full_ms)) <= REALTIME_BUDGET_MS,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    info = device_info()
    print(f"device: {info['device']}")
    print(
        f"  index={info['device_index']}/{info['device_count']}  "
        f"sm={info['compute_capability']}  mem={info['total_memory_mib']} MiB"
    )
    print(
        f"  torch={info['torch_version']}  tensorrt={info['tensorrt_version']}  "
        f"driver={info['driver_version']}  cuda={info['cuda_runtime']}"
    )

    ensure_onnx(ONNX_PATH)
    export_fixed_trt_engine(ONNX_PATH, FIXED_ENGINE_PATH, FIXED_BATCH)

    model = TRTVisionEncoder(FIXED_ENGINE_PATH)
    text_bank = make_text_bank()
    frame = load_image(SAMPLE_IMAGE)

    print(
        f"================================== Start Fixed Engine: batch={FIXED_BATCH}, {N_INFER}x chained =================================="
    )
    r = measure(model, text_bank, frame)
    df = pd.DataFrame([r])[
        [
            "fixed_batch",
            "n_infer",
            "channels_per_cycle",
            "single_batch_ms_mean",
            "inference_only_ms_mean",
            "inference_only_ms_p95",
            "full_cycle_ms_mean",
            "full_cycle_ms_p95",
            "inference_only_meets_500ms",
            "full_cycle_meets_500ms",
        ]
    ]
    print(df.to_string(index=False))
    print(
        f"================================== End Fixed Engine: batch={FIXED_BATCH}, {N_INFER}x chained =================================="
    )

    payload = {
        **info,
        "precision": "fp16",
        "input_shape": list(INPUT_SIZE),
        "engine_kind": "fixed (min=opt=max=fixed_batch)",
        "fixed_batch": FIXED_BATCH,
        "n_infer_per_cycle": N_INFER,
        "target_channels": TARGET_CHANNELS,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "engine_path": FIXED_ENGINE_PATH,
        "sample_image": SAMPLE_IMAGE,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "result": r,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")

    msg = "PASS" if r["inference_only_meets_500ms"] else "FAIL"
    print(
        f"{msg} (inference-only): {N_INFER}x batch={FIXED_BATCH} = {r['inference_only_ms_mean']:.2f} ms "
        f"(budget {REALTIME_BUDGET_MS:.0f} ms)"
    )
    print(
        f"  vs dynamic engine equivalent measured earlier: 324.6 ms (compare with {r['inference_only_ms_mean']:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
