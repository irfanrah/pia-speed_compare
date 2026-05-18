"""pia.ai.exports — model export utilities.

`yolo` exports depend on `onnx` and `ultralytics`. Importing this package
must not pull those in unless a caller actually uses the YOLO export path.
`trt.onnx2trt` is dependency-light and exposed eagerly so existing imports
like `from pia.ai.exports import onnx2trt` keep working unchanged.
"""

from .trt import onnx2trt


def __getattr__(name):
    if name == "yolo":
        from . import yolo as _yolo

        return _yolo
    if name == "yolo2onnx":
        from .yolo import yolo2onnx

        return yolo2onnx
    if name == "yolo2trt":
        from .yolo import yolo2trt

        return yolo2trt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["yolo", "onnx2trt", "yolo2onnx", "yolo2trt"]
