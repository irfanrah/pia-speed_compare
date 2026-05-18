from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.glenc_harness.config import (
    OD_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
    CLS_CONFIDENCE_THRESHOLD,
)


class HarnessWithClsModel(CategoryBase):
    # NOTE: 현재 미사용 — service._detect는 config 상수만 참조한다.
    # 사용자 파라미터 전달 경로가 정비되면 활성화 예정.
    od_threshold: float = OD_CONFIDENCE_THRESHOLD
    iou_threshold: float = OD_NMS_THRESHOLD
    cls_threshold: float = CLS_CONFIDENCE_THRESHOLD
