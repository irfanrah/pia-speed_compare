from pydantic import Field

from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.glenc_workinghigh.config import OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD


class WorkinghighModel(CategoryBase):
    confidence_threshold: float = Field(
        default=OD_CONFIDENCE_THRESHOLD, alias="confidenceThreshold"
    )
    nms_threshold: float = Field(default=OD_NMS_THRESHOLD, alias="nmsThreshold")

    model_config = {"populate_by_name": True}
