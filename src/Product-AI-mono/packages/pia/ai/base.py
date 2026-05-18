from abc import ABC, abstractmethod
from typing import Literal, Union, final

import torch


class PiaONNXConfigBase(ABC):
    @abstractmethod
    def __init__(self, output_dir: str, opset: int = 14, half: bool = False):
        self.output_dir = output_dir
        self.opset = opset
        self.half = half


class PiaConfigBase(ABC):
    @abstractmethod
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_half_precision: bool = False,
        tile_config: Literal["L", "M", "S"] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.use_half_precision = use_half_precision
        self.tile_config = tile_config


class PiaModelBase(ABC):
    @abstractmethod
    def __init__(self, config: PiaConfigBase) -> None:
        super().__init__(config)
        self.config = config

    @final
    def __load_model__(self):
        model = self._load_model()
        if self.config.use_half_precision:
            return model.half()
        return model

    @final
    @torch.no_grad
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def forward(self, *args):
        pass


class PiaFactoryBase(ABC):
    @abstractmethod
    def load(
        self, target_model: Union[str, int], config: PiaConfigBase
    ) -> PiaModelBase:
        raise NotImplementedError("`load()` function is not implemented properly!")

    # @abstractmethod
    # def get_available_models(self) -> Mapping[str, List[str]]:
    #     raise NotImplementedError(
    #         "`get_available_models()` is not implemented properly!")
