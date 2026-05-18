from typing import Union

from pia.ai.base import PiaFactoryBase

from .base import T2VRetConfig, T2VRetModelBase

# Lazy: importing this factory must not pull in transformers/einops/...
# (clip_vip needs transformers, PE needs einops, qwen3 needs both).
_MODEL_KEY = {
    0: "clip4clip",
    "clip4clip": "clip4clip",
    1: "clip-vip",
    "clip-vip": "clip-vip",
    2: "PerceptionEncoder",
    "PerceptionEncoder": "PerceptionEncoder",
    3: "Qwen3VLEmbedding",
    "Qwen3VLEmbedding": "Qwen3VLEmbedding",
}


def _load_model(key: str):
    if key == "clip4clip":
        from .models.clip4clip.main import Clip4Clip

        return Clip4Clip
    if key == "clip-vip":
        from .models.clip_vip.main import CLIPVIP

        return CLIPVIP
    if key == "PerceptionEncoder":
        from .models.PE.main import PerceptionEncoder

        return PerceptionEncoder
    if key == "Qwen3VLEmbedding":
        from .models.qwen3_vl_embedding.main import Qwen3VLEmbedding

        return Qwen3VLEmbedding
    raise KeyError(f"Unknown T2VRet model key: {key!r}")


SUPPORTED_MODELS = _MODEL_KEY


class T2VRetFactory(PiaFactoryBase):
    def __init__(self, target_model: Union[str, int], config: T2VRetConfig) -> None:
        if target_model not in _MODEL_KEY:
            raise KeyError(f"Unknown T2VRet model: {target_model!r}")
        model_cls = _load_model(_MODEL_KEY[target_model])
        self._model = model_cls(config=config)

    def load(self) -> T2VRetModelBase:
        return self._model
