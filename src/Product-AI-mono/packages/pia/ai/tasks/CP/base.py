from abc import abstractmethod
from typing import Union

import numpy as np
from pia.ai.base import PiaConfigBase, PiaModelBase


class CPONNXConfig(PiaConfigBase):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_half_precision: bool = False,
        window_size=224,
        stride=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.window_size = window_size
        self.stride = stride
        self.mean = mean
        self.std = std
        self.use_half_precision = use_half_precision

        super().__init__(model_path, device)


class CPModelBase(PiaModelBase):
    @abstractmethod
    # @torch.no_grad - overriding
    def predict(self, image: Union[str, np.ndarray]) -> float:
        pass
