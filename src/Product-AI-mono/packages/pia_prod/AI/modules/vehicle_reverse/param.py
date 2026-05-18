from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.vehicle_reverse.config import OD_CONFIDENCE_THRESHOLD, IOU_THRESHOLD


class VehicleReverseModel(CategoryBase):
    """User parameter defaults for vehicle reverse detection."""

    od_threshold: float = OD_CONFIDENCE_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
