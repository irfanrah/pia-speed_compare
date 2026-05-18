from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.yonsei_walljump.config import (
    OD_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
    CLS_CONFIDENCE_THRESHOLD,
)


class WalljumpModel(CategoryBase):
    od_threshold: float = OD_CONFIDENCE_THRESHOLD
    iou_threshold: float = OD_NMS_THRESHOLD
    cls_threshold: float = CLS_CONFIDENCE_THRESHOLD
