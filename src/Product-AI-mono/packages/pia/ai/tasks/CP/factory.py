from typing import Union

from pia.ai.base import PiaFactoryBase

from .base import CPModelBase, CPONNXConfig

# Lazy: importing this factory must not pull in onnxruntime.
_MODEL_KEY = {
    0: "clip_ebc_onnx",
    "clip_ebc_onnx": "clip_ebc_onnx",
}


def _load_model(key: str):
    if key == "clip_ebc_onnx":
        from .models.clip_ebc.clip_ebc import ClipEBCOnnx

        return ClipEBCOnnx
    raise KeyError(f"Unknown CP model key: {key!r}")


SUPPORTED_MODELS = _MODEL_KEY


class CpFactory(PiaFactoryBase):
    def __init__(self, target_model: Union[str, int], config: CPONNXConfig) -> None:
        if target_model not in _MODEL_KEY:
            raise KeyError(f"Unknown CP model: {target_model!r}")
        model_cls = _load_model(_MODEL_KEY[target_model])
        self._model = model_cls(config=config)

    def load(self) -> CPModelBase:
        return self._model
