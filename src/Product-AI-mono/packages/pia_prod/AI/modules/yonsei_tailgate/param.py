from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.yonsei_tailgate.config import (
    TAILGATE_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
)


class TailgateModel(CategoryBase):
    od_threshold: float = TAILGATE_CONFIDENCE_THRESHOLD
    iou_threshold: float = OD_NMS_THRESHOLD
