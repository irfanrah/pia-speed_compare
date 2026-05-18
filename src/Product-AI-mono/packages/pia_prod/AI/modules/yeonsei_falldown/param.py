from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.yeonsei_falldown.config import (
    ANGLE_THRESHOLD,
    BBOX_KEEP_HEAD_FOOT_RATIO,
    BBOX_CUTTING_MARGIN_RATIO,
)


class FalldownModel(CategoryBase):
    angle_threshold: int = ANGLE_THRESHOLD
    bbox_keep_head_foot_ratio: float = BBOX_KEEP_HEAD_FOOT_RATIO
    bbox_cutting_margin_ratio: float = BBOX_CUTTING_MARGIN_RATIO
