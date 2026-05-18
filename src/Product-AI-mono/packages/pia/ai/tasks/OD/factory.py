from typing import Union

from pia.ai.base import PiaFactoryBase

from .base import ODConfig, ODModelBase

# Lazy: importing this factory must not pull in ultralytics (via yolov8.main).
_MODEL_KEY = {
    0: "yolov8",
    "yolov8": "yolov8",
}


def _load_model(key: str):
    if key == "yolov8":
        from .models.yolov8.main import YOLOv8

        return YOLOv8
    raise KeyError(f"Unknown OD model key: {key!r}")


SUPPORTED_MODELS = _MODEL_KEY


class ODFactory(PiaFactoryBase):
    def __init__(self, target_model: Union[str, int], config: ODConfig) -> None:
        if target_model not in _MODEL_KEY:
            raise KeyError(f"Unknown OD model: {target_model!r}")
        model_cls = _load_model(_MODEL_KEY[target_model])
        self._model = model_cls(config=config)

    def load(self) -> ODModelBase:
        return self._model
