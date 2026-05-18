from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.yeonsei_smoke.config import (
    LOWER_HSV,
    UPPER_HSV,
    BBOX_MIN_AREA,
    SMOKE_THRESHOLD,
)


class SmokeModel(CategoryBase):
    lower_hsv: tuple = LOWER_HSV
    upper_hsv: tuple = UPPER_HSV
    bbox_min_area: int = BBOX_MIN_AREA
    cls_threshold: float = SMOKE_THRESHOLD
