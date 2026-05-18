from .device import load_model_backend


def __getattr__(name):
    """Lazy import for heavy export utilities (onnx, ultralytics).

    onnx2trt, yolo2onnx, yolo2trt are only needed when explicitly converting
    models. Eagerly importing them pulls in onnx and ultralytics, which are
    unnecessary for modules like perception_encoder that use TensorRT directly.
    """
    _lazy_exports = {
        "onnx2trt": (".exports.trt", "onnx2trt"),
        "yolo2onnx": (".exports.yolo", "yolo2onnx"),
        "yolo2trt": (".exports.yolo", "yolo2trt"),
    }
    if name in _lazy_exports:
        module_path, attr = _lazy_exports[name]
        import importlib

        mod = importlib.import_module(module_path, __package__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_model_backend",
    "onnx2trt",
    "yolo2onnx",
    "yolo2trt",
]
