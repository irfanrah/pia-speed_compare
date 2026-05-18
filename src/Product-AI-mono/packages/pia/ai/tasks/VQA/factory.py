from typing import Union

from pia.ai.base import PiaFactoryBase

from .base import VQAConfig, VQAModelBase

# Lazy: importing this factory must not pull in transformers / tensorrt_llm /
# the heavy InternVL3 import chain.
_MODEL_KEY = {
    0: "internvl3",
    "internvl3": "internvl3",
    1: "internvl3trt",
    "internvl3trt": "internvl3trt",
    2: "internvl3trt_llm",
    "internvl3trt_llm": "internvl3trt_llm",
}


def _load_model(key: str):
    if key == "internvl3":
        from .models.internVL3.main import InternVL3

        return InternVL3
    if key == "internvl3trt":
        from .models.internVL3trt.main import InternVL3trt

        return InternVL3trt
    if key == "internvl3trt_llm":
        from .models.internVL3trt_llm.main import InternVL3trt_llm

        return InternVL3trt_llm
    raise KeyError(f"Unknown VQA model key: {key!r}")


SUPPORTED_MODELS = _MODEL_KEY


class VQAFactory(PiaFactoryBase):
    def __init__(self, target_model: Union[str, int], config: VQAConfig) -> None:
        if isinstance(target_model, str):
            target_model = target_model.lower()
        if target_model not in _MODEL_KEY:
            raise KeyError(f"Unknown VQA model: {target_model!r}")
        model_cls = _load_model(_MODEL_KEY[target_model])
        self._model = model_cls(config=config)

    def load(self) -> VQAModelBase:
        return self._model
