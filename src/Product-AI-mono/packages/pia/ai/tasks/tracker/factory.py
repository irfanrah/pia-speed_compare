from typing import Union

from pia.ai.base import PiaFactoryBase

from .base import TrackerBase, TrackerConfig

# Lazy: importing this factory must not pull in filterpy (Kalman filter).
_MODEL_KEY = {
    0: "sort",
    "sort": "sort",
}


def _load_model(key: str):
    if key == "sort":
        from .models.sort.sort import Sort

        return Sort
    raise KeyError(f"Unknown tracker model key: {key!r}")


SUPPORTED_MODELS = _MODEL_KEY


class TrackerFactory(PiaFactoryBase):
    def __init__(self, target_model: Union[str, int], config: TrackerConfig) -> None:
        if target_model not in _MODEL_KEY:
            raise KeyError(f"Unknown tracker model: {target_model!r}")
        model_cls = _load_model(_MODEL_KEY[target_model])
        self._model = model_cls(**config.__dict__)

    def load(self) -> TrackerBase:
        return self._model
