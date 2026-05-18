from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.khonkaen_weapon.config import (
    OD_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
    WEAPON_DETECTION_CONFIDENCE_THRESHOLD,
)


class WeaponModel(CategoryBase):
    od_threshold: float = OD_CONFIDENCE_THRESHOLD
    iou_threshold: float = OD_NMS_THRESHOLD
    sgie_threshold: float = WEAPON_DETECTION_CONFIDENCE_THRESHOLD
