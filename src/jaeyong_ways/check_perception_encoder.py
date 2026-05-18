"""PE-Core-L14-336 inference time checker (TRT FP16).

Standalone reimplementation of
``Research-AI-mono/prod_ai_inference_time_checker/check_perception_encoder.py``
that does not depend on ``pia_prod`` / ``pia`` packages. Runs inside the
``Product-AI-mono`` conda env.

Pipeline mirrors the upstream stages:

    model_preprocess  →  model_inference  →  postprocess_logic

For each batch size 1..MAX_BATCH_SIZE we time the three stages over many
iterations, print a pandas DataFrame in the upstream style, and emit a result
JSON containing the active CUDA device and a realtime verdict.
"""

import json
import os
import subprocess
import sys
import time
from statistics import median
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorrt as trt
import torch
from huggingface_hub import hf_hub_download

from src.preprocess import load_image, preprocess

# ──────────────────────────────── config ────────────────────────────────

HF_REPO_ID = "PIA-SPACE-LAB/PE-Core-L14-336"
HF_ONNX_FILENAME = "onnx/PE-Core-L14-336_vision_dynamic.onnx"

ONNX_PATH = "assets/model/PE-Core-L14-336_vision_dynamic.onnx"
ENGINE_PATH = "assets/model/PE-Core-L14-336_vision_dynamic.engine"
SAMPLE_IMAGE = "assets/images/dog.jpg"
RESULTS_JSON = "results/pe_check_perception_encoder.json"

IMG_SIZE = (336, 336)
INPUT_SIZE = (3, *IMG_SIZE)

MAX_BATCH_SIZE = 16
TRT_PROFILE = {"min": 1, "opt": 16, "max": max(32, MAX_BATCH_SIZE)}

WARMUP_ITERS = 20
MEASURE_ITERS = 50

REALTIME_BUDGET_MS = 500.0
TARGET_CHANNELS = 12

# Pretend text-embedding bank used by the postprocess stage. PE-Core-L14-336
# emits 1024-dim visual features; cosine sim against a small bank approximates
# the alarm_event_manager step in upstream's perception_encoder_service.
TEXT_BANK_DIM = 1024
TEXT_BANK_SIZE = 16

# ─────────────────────────── model preparation ──────────────────────────


def ensure_onnx(local_path: str = ONNX_PATH) -> str:
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"Downloading {HF_ONNX_FILENAME} from {HF_REPO_ID} ...")
    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ONNX_FILENAME)
    if os.path.abspath(cached) != os.path.abspath(local_path):
        import shutil

        shutil.copy(cached, local_path)
    print(f"ONNX ready at {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB)")
    return local_path


def export_trt_engine(
    onnx_file: str = ONNX_PATH,
    save_file_path: str = ENGINE_PATH,
    input_size=INPUT_SIZE,
    min_batch: int = TRT_PROFILE["min"],
    opt_batch: int = TRT_PROFILE["opt"],
    max_batch: int = TRT_PROFILE["max"],
) -> str:
    if os.path.exists(save_file_path):
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
        min=(min_batch, *input_size),
        opt=(opt_batch, *input_size),
        max=(max_batch, *input_size),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TRT engine")
    with open(save_file_path, "wb") as f:
        f.write(serialized)
    print(f"Engine saved at {save_file_path} ({os.path.getsize(save_file_path) / 1e6:.1f} MB)")
    return save_file_path


# ───────────────────────────── TRT inference ────────────────────────────

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def _run(context):
    if hasattr(context, "execute_v3"):
        return context.execute_v3()
    if hasattr(context, "execute_async_v3"):
        return context.execute_async_v3(stream_handle=0)
    raise RuntimeError("TRT v3 execution API unavailable")


class TRTVisionEncoder:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.in_name = next(
            n for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.out_names = [
            n for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        self.out_dtypes = {
            n: _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)] for n in self.out_names
        }
        self._out_shape: tuple = ()
        self.outputs: Dict[str, torch.Tensor] = {}

    def __call__(self, image_cuda: torch.Tensor) -> torch.Tensor:
        self.context.set_input_shape(self.in_name, tuple(image_cuda.shape))
        for n in self.out_names:
            dims = self.context.get_tensor_shape(n)
            shape = tuple([int(image_cuda.shape[0])] + [int(d) for d in dims[1:]])
            if self.outputs.get(n) is None or tuple(self.outputs[n].shape) != shape:
                self.outputs[n] = torch.empty(shape, dtype=self.out_dtypes[n], device="cuda")

        self.context.set_tensor_address(self.in_name, int(image_cuda.data_ptr()))
        for n in self.out_names:
            self.context.set_tensor_address(n, int(self.outputs[n].data_ptr()))

        if not _run(self.context):
            raise RuntimeError("TRT execute failed")
        return self.outputs[self.out_names[0]]


# ───────────────────────────── postprocess ──────────────────────────────


def make_text_bank(device: str = "cuda") -> torch.Tensor:
    """Random unit-norm text-feature bank, frozen across the run."""
    g = torch.Generator(device=device).manual_seed(0)
    bank = torch.randn(TEXT_BANK_SIZE, TEXT_BANK_DIM, device=device, generator=g)
    return bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.inference_mode()
def postprocess(visual: torch.Tensor, text_bank: torch.Tensor) -> torch.Tensor:
    """Mimic alarm_event_manager: L2-normalize visual, cosine sim vs bank, top-1."""
    v = visual.float()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = v @ text_bank.T
    return sim.argmax(dim=-1)


# ─────────────────────────────── checker ────────────────────────────────


def device_info() -> dict:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        driver = None
    return {
        "device": torch.cuda.get_device_name(idx),
        "device_index": idx,
        "device_count": torch.cuda.device_count(),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mib": int(props.total_memory / 1024 / 1024),
        "driver_version": driver,
        "torch_version": torch.__version__,
        "tensorrt_version": trt.__version__,
        "cuda_runtime": torch.version.cuda,
    }


@torch.inference_mode()
def test_pe_batches(
    image_path: str = SAMPLE_IMAGE,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> List[dict]:
    model = TRTVisionEncoder(ENGINE_PATH)
    text_bank = make_text_bank()
    frame = load_image(image_path)

    rows: List[dict] = []
    for now_batch in range(1, max_batch_size + 1):
        print(
            f"================================== Start Test Batch Size: {now_batch} =================================="
        )

        batches = [frame] * now_batch
        for _ in range(WARMUP_ITERS):
            inp = preprocess(batches, resize_size=IMG_SIZE, device="cuda")
            feats = model(inp)
            _ = postprocess(feats, text_bank)
        torch.cuda.synchronize()

        time_dict = {"total": [], "model_preprocess": [], "model_inference": [], "postprocess_logic": []}
        for _ in range(MEASURE_ITERS):
            t0 = time.perf_counter()
            inp = preprocess(batches, resize_size=IMG_SIZE, device="cuda")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            feats = model(inp)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            _ = postprocess(feats, text_bank)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            time_dict["total"].append((t3 - t0) * 1000)
            time_dict["model_preprocess"].append((t1 - t0) * 1000)
            time_dict["model_inference"].append((t2 - t1) * 1000)
            time_dict["postprocess_logic"].append((t3 - t2) * 1000)

        averages = {k: sum(v) / len(v) for k, v in time_dict.items()}
        df = pd.DataFrame([averages])
        print(df.to_string(index=False))
        print(
            f"================================== End Test Batch Size: {now_batch} =================================="
        )

        rows.append(
            {
                "batch": now_batch,
                **{k: round(averages[k], 3) for k in averages},
                "p95_total_ms": round(_percentile(time_dict["total"], 95), 3),
                "images_per_sec": round(now_batch * 1000.0 / averages["total"], 2),
                "meets_realtime_500ms": averages["total"] <= REALTIME_BUDGET_MS,
            }
        )

    return rows


def _percentile(values, p):
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


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
    export_trt_engine(ONNX_PATH, ENGINE_PATH)

    rows = test_pe_batches(SAMPLE_IMAGE, MAX_BATCH_SIZE)

    target = next((r for r in rows if r["batch"] == TARGET_CHANNELS), None)
    target_pass = bool(target and target["meets_realtime_500ms"])

    payload = {
        **info,
        "precision": "fp16",
        "input_shape": list(INPUT_SIZE),
        "engine_profile": TRT_PROFILE,
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "target_channels": TARGET_CHANNELS,
        "sample_image": SAMPLE_IMAGE,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "results": rows,
        "target_result": target,
        "pass": target_pass,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")

    msg = "PASS" if target_pass else "FAIL"
    if target:
        print(
            f"{msg}: batch={TARGET_CHANNELS} total={target['total']:.2f} ms "
            f"(budget {REALTIME_BUDGET_MS:.0f} ms)"
        )
    return 0 if target_pass else 1


if __name__ == "__main__":
    sys.exit(main())
