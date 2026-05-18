from collections import defaultdict
from typing import List
import numpy as np
from pia.vision.roi.roi_manager import (
    ROIManagerBase,
    calc_expand_coord,
    cv_crop_region,
    calc_letterbox_parameter,
)
from pia_prod.AI.modules.ax_harness.config import OD_INPUT_SIZE


class HarnessRoIManager(ROIManagerBase):
    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(dict)

    def process_batches_with_roi(
        self, batches: List[np.ndarray], stream_ids: List[str], rois: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        사용자가 입력한 ROI에 비례해서 일정크기 이상의 부분을 확대하여 Crop하여 반환한다
        """
        result = []
        for i, (roi, stream_id) in enumerate(zip(rois, stream_ids)):
            w, h = batches[i].shape[1], batches[i].shape[0]
            # wrapping
            roi = [[0, 0], [0, h], [w, h], [w, 0]] if len(roi) == 0 else self.clip_roi(roi, w, h)
            
            # ROI 확장 및 직사각형화
            expanded_roi = calc_expand_coord(
                roi=roi,
                frame_wh=(w, h),
            )

            # Crop 및 마스킹 처리
            cropped_batch = cv_crop_region(batches[i], roi)
            result.append(cropped_batch)
            self.roi_dict[stream_id]["after_letterbox_calc_origin_roi"] = (
                self.find_after_letterbox_calc_origin_roi(roi, expanded_roi)
            )
        return result

    @staticmethod
    def find_after_letterbox_calc_origin_roi(origin_roi, expand_roi):
        re_calc_origin_roi = origin_roi - expand_roi[0]
        expanded_roi_shape = list(reversed(list(expand_roi[2] - expand_roi[0])))
        r, pad = calc_letterbox_parameter(
            source_shape=expanded_roi_shape, target_shape=OD_INPUT_SIZE
        )
        after_letterbox_calc_origin_roi = ((re_calc_origin_roi * r) + np.array(pad)).astype(
            np.int32
        )

        return after_letterbox_calc_origin_roi
