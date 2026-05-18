from pia.ai.base import PiaFactoryBase

from .base import FilterConfigBase
from .filters import (
    AverageFilter,
    EWMAFilter,
    MovingAverageFilter,
    StreamingSavitzkyGolayFilter,
)

AVALIABLE_FILTERS = {
    0: AverageFilter,
    "avg_filter": AverageFilter,
    1: MovingAverageFilter,
    "moving_avg": MovingAverageFilter,
    2: StreamingSavitzkyGolayFilter,  # polyorder config 설정 필요
    "savitzkygolay": StreamingSavitzkyGolayFilter,  # polyorder config 설정 필요
    3: EWMAFilter,
    "EMA": EWMAFilter,
    # 4 :
}


class FilterFactory(PiaFactoryBase):
    @staticmethod
    def get_avaliable_filters():
        return [q for q in AVALIABLE_FILTERS.keys() if isinstance(q, str)]

    @staticmethod
    def load(config: FilterConfigBase):
        return AVALIABLE_FILTERS[config.filter_type](config)
