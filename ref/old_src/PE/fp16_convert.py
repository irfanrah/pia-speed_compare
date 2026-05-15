import os

import onnx
from onnxconverter_common.float16 import convert_float_to_float16


def ensure_fp16_onnx(src_path: str, dst_path: str) -> str:
    """Convert an FP32 ONNX to FP16 weights, keeping IO as FP32.

    Skips if dst already exists. Uses external-data save format so the result
    survives the 2 GB protobuf limit even if the model is large.
    """
    if os.path.exists(dst_path):
        print(f"FP16 ONNX already present at {dst_path}, skipping convert.")
        return dst_path
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source ONNX not found: {src_path}")

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    print(f"Converting weights to FP16: {src_path} -> {dst_path}")

    model = onnx.load(src_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, dst_path)

    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"FP16 ONNX written ({size_mb:.2f} MB).")
    return dst_path
