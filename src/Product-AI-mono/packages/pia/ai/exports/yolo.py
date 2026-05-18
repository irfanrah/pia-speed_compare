import argparse
from pathlib import Path
from typing import Union

import onnx

from ..exports.trt import onnx2trt

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def yolo2onnx(
    model_path: Union[str, Path],
    opset:int = 17,
    overwrite: bool = False,
    half:bool = True
):
    """Convert a YOLO model (.pt) to ONNX model format."""
    model_path = Path(model_path)
    onnx_path = model_path.with_suffix(".onnx")

    if YOLO is None:  # pragma: no cover - handled in tests via mocks
        raise ImportError("ultralytics is required for yolo2onnx conversion")

    if not overwrite and onnx_path.exists():
        print("ONNX model already exists. Skipping conversion.")
        return onnx_path

    pytorch_model = YOLO(str(model_path))

    pytorch_model.export(
        format="onnx", dynamic=True, opset=opset, dynamo=False
    )

    if half :
        from onnxconverter_common import float16
        model = onnx.load(onnx_path)
        # Convert the model to float16
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, onnx_path)
    return onnx_path


def yolo2trt(
    model_path: Union[str, Path],
    batch_size: int = 1,
    device: int = 0,
    overwrite: bool = False,
    half: bool = False,
):
    """Convert a YOLO model (.pt) or ONNX model to TensorRT format."""
    model_path = Path(model_path)

    if model_path.suffix == ".onnx":
        return onnx2trt(model_path, device=device, fp16=half, overwrite=overwrite)

    if YOLO is None:  # pragma: no cover - handled in tests via mocks
        raise ImportError("ultralytics is required for yolo2trt conversion")

    def is_available_trt_file(path: Path) -> bool:
        trt_name_list = [
            str(path).replace(".pt", ".engine"),
            str(path).replace(".pt", "_dynamic.engine"),
            str(path).replace(".pt", "_dynamic_fp16.engine"),
            str(path) + ".engine",
            str(path) + "_dynamic.engine",
            str(path) + "_dynamic_fp16.engine",
        ]
        for trt_name in trt_name_list:
            if Path(trt_name).exists():
                print(f"TensorRT model already exists: {trt_name}")
                return True
        return False

    if not overwrite and is_available_trt_file(model_path):
        print("TensorRT model already exists. Skipping conversion.")
        return None

    pytorch_model = YOLO(str(model_path)).cuda(device)

    pytorch_model.export(
        format="engine", half=half, dynamic=True, device=device, batch=batch_size
    )
    return model_path.with_suffix(".engine")


def _parse_args():  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Convert models to TensorRT format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (.pt or .onnx)")
    parser.add_argument("--num_max_cameras", type=int, default=32, help="Maximum number of cameras")
    parser.add_argument(
        "--num_max_person_per_camera", type=int, default=8, help="Max people per camera"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing models")
    parser.add_argument("--half", action="store_true", help="Use FP16 (half precision)")
    parser.add_argument(
        "--to_onnx",
        action="store_true",
        help="Only export the YOLO model to ONNX and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - CLI use
    args = _parse_args()
    if args.to_onnx:
        yolo2onnx(
            model_path=args.model_path,
            device=args.device,
            overwrite=args.overwrite,
        )
    else:
        yolo2trt(
            model_path=args.model_path,
            num_max_cameras=args.num_max_cameras,
            num_max_person_per_camera=args.num_max_person_per_camera,
            device=args.device,
            overwrite=args.overwrite,
            half=args.half,
        )
    print("Conversion completed.")
