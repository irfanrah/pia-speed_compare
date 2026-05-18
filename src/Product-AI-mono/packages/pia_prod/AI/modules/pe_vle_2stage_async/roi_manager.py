"""
PE-VLE ROI manager — local copy of pe_vqa_2stage.PeVqa2StageRoIManager.
Falldown is the only category that uses ROI cropping (PE convention).
"""

from collections import defaultdict
from typing import List

import numpy as np
import torch

from pia.vision.roi.roi_manager import ROIManagerBase, batch_crop_region
from pia_prod.AI.global_config import (
    CAMERA_ID_KEY,
    DIVIDED_ROI_KEY,
    EXPANDED_ROI_KEY,
    RET_EVENT_KEY,
    ROI_KEY,
    ROI_POLYGON_COORDINATES_KEY,
    USER_PARAM_KEY,
)
from pia_prod.AI.modules.pe_vle_2stage_async.config import (
    ALL_CATEGORIES,
    DEVICE,
    FALLDOWN_CATEGORY,
)


class PeVleRoIManager(ROIManagerBase):
    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(lambda: defaultdict(dict))
        self.category_list = ALL_CATEGORIES
        self.roi_category_list = FALLDOWN_CATEGORY  # Only falldown uses ROI

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        raw_roi = roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
        if len(raw_roi) != 0:
            coordinate = self.get_pair_list(raw_roi)
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [
                [0, 0],
                [0, image_wh[1]],
                [image_wh[0], image_wh[1]],
                [image_wh[0], 0],
            ]
        after_letterbox_calc_origin_roi_list = []
        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = raw_roi
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = coordinate
        self.roi_dict[camera_id][DIVIDED_ROI_KEY] = after_letterbox_calc_origin_roi_list

    def get_roi_info(self, camera_id: str, roi_raw_info: dict, image_wh: tuple) -> dict:
        if camera_id not in self.roi_dict:
            self.add_roi(camera_id=camera_id, roi_raw_info=roi_raw_info, image_wh=image_wh)
        if (
            roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
            != self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY]
        ):
            self.add_roi(camera_id=camera_id, roi_raw_info=roi_raw_info, image_wh=image_wh)

        return self.roi_dict[camera_id]

    def process_batches_with_roi(
        self, batches: List[np.array], user_params: List
    ) -> List[np.array]:
        regions = []
        gpu_batches = []

        for idx, (batch, user_param) in enumerate(zip(batches, user_params)):
            ret_event = user_param[USER_PARAM_KEY][RET_EVENT_KEY]

            found_key = None
            for key in self.roi_category_list:
                if key in ret_event:
                    found_key = key
                    break

            if found_key is None:
                roi_dict = self.get_roi_info(
                    camera_id=user_param[USER_PARAM_KEY][CAMERA_ID_KEY],
                    roi_raw_info={ROI_KEY: {ROI_POLYGON_COORDINATES_KEY: []}},
                    image_wh=tuple(reversed(batch.shape[:2])),
                )
            else:
                roi_dict = self.get_roi_info(
                    camera_id=user_param[USER_PARAM_KEY][CAMERA_ID_KEY],
                    roi_raw_info=ret_event[found_key],
                    image_wh=tuple(reversed(batch.shape[:2])),
                )
            gpu_batch = (
                torch.from_numpy(batch).to(DEVICE) if isinstance(batch, np.ndarray) else batch
            )

            gpu_batches.append(gpu_batch)
            regions.append(roi_dict[EXPANDED_ROI_KEY])

        cropped_images = batch_crop_region(gpu_batches, regions)
        return cropped_images
