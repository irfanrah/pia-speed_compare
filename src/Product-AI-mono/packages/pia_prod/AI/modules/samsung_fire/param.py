# AI/modules/samsung_fire/fire_param.py

from typing import Optional, Tuple
from pydantic import BaseModel

from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.modules.samsung_fire.config import (
    LOWER_HSV,
    UPPER_HSV,
    LOWER_YCrCb,
    UPPER_YCrCb,
    MOTION_THRESH_MIN,
    MOTION_THRESH_MAX,
    MIN_MOTION_RATIO,
    ALARM_TRUE_RATIO_THRESH,
    TRACK_IOU_THRESH,
    COLOR_FILTER_KERNEL_SIZE,
    MOTION_FILTER_KERNEL_SIZE,
    ROI_RESIZE_SIZE,
    BBOX_MIN_AREA,
    MAX_PARTICIPATE_BBOX_NUM,
    FIRE_BOX_EXPAND_RATIO,
)


class FireModel(BaseModel):
    name: str = "fire"
    incidentThresholdSecond: int
    incidentTimeoutSecond: int
    roi: Optional[ROIModel] = ROIModel()

    lower_hsv: Tuple[int, int, int] = LOWER_HSV
    upper_hsv: Tuple[int, int, int] = UPPER_HSV
    lower_ycc: Tuple[int, int, int] = LOWER_YCrCb
    upper_ycc: Tuple[int, int, int] = UPPER_YCrCb

    motionThreshMin: int = MOTION_THRESH_MIN
    motionThreshMax: int = MOTION_THRESH_MAX
    minMotionRatio: float = MIN_MOTION_RATIO

    colorFilterKernelSize: Tuple[int, int] = COLOR_FILTER_KERNEL_SIZE
    motionFilterKernelSize: Tuple[int, int] = MOTION_FILTER_KERNEL_SIZE

    fireAlarmRatioThreshold: float = ALARM_TRUE_RATIO_THRESH
    iouThreshold: float = TRACK_IOU_THRESH

    bbox_min_area: int = BBOX_MIN_AREA
    maxParticipateBboxNum: int = MAX_PARTICIPATE_BBOX_NUM
    fireBoxExpandRatio: float = FIRE_BOX_EXPAND_RATIO

    resizeSize: Tuple[int, int] = ROI_RESIZE_SIZE
