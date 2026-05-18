from abc import abstractmethod
from typing import List, final


class FilterConfigBase:
    @abstractmethod
    def __init__(
        self,
        filter_type: str,
        window_length: int = None,
        alpha_weight: float = None,
        **keyword
    ) -> None:
        self.filter_type = filter_type
        self.window_length = window_length
        self.alpha_weight = alpha_weight
        self.keyword = keyword


class FilterBase:
    def __init__(self, config: FilterConfigBase) -> None:
        self.pre_v = 0
        self.config = config

    def get_name(self):
        return self.config.filter_type

    @abstractmethod
    def filter(self, x: float):
        pass

    @final
    def apply_filter(self, obs):
        return self.filter(obs=obs)

    @final
    def scaled_output(self, obs):
        v, pre_v = self.apply_filter(obs)
        diff = abs(pre_v - v)
        self.pre_avg = v
        return diff

    @final
    def __call__(self, *args, **keyward) -> List[float]:
        v = self.filter(keyward["obs"])
        pre_avg = self.pre_v
        self.pre_v = v
        return v, pre_avg
