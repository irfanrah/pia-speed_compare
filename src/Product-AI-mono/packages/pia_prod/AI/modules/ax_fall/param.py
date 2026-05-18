from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.ax_fall.config import OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD


class FallModel(CategoryBase):
    confidence_threashold: float = OD_CONFIDENCE_THRESHOLD
    nms_threshold: float = OD_NMS_THRESHOLD
