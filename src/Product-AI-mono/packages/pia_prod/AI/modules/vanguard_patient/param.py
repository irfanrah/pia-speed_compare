from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.vanguard_patient.config import OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, CLS_CONFIDENCE_THRESHOLD


class PatientModel(CategoryBase):
    confidence_threashold: float = OD_CONFIDENCE_THRESHOLD
    nms_threshold: float = OD_NMS_THRESHOLD
    cls_confidence_threashold: float = CLS_CONFIDENCE_THRESHOLD
