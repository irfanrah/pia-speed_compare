from collections import defaultdict
from typing import List

import numpy as np
import torch

from pia_prod.AI.global_config import (
    CAMERA_ID_KEY,
    EXPANDED_ROI_KEY,
    RET_EVENT_KEY,
    ROI_KEY,
    ROI_POLYGON_COORDINATES_KEY,
    USER_PARAM_KEY,
)
from pia.vision.roi.roi_manager import ROIManagerBase, batch_crop_region
from pia_prod.AI.modules.qwen3vle_trt.config import ALL_CATEGORIES, DEVICE


class Qwen3VLETrtRoIManager(ROIManagerBase):
    """
    Multi-category RoI Manager for the ONNX/TRT variant.
    - Manages RoI information for all supported categories.
    - Caches latest RoI coordinates per camera.
    - Falls back to full-frame when no RoI is defined.
    """

    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(lambda: defaultdict(dict))
        self.roi_category_list = ALL_CATEGORIES

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple) -> None:
        raw_roi = roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
        if len(raw_roi) != 0:
            coordinate = self.get_pair_list(raw_roi)
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = np.array(
                [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]],
                dtype=np.int32,
            )
        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = raw_roi
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = np.asarray(coordinate, dtype=np.int32)

    def get_roi_info(self, camera_id: str, roi_raw_info: dict, image_wh: tuple) -> dict:
        if camera_id not in self.roi_dict:
            self.add_roi(camera_id=camera_id, roi_raw_info=roi_raw_info, image_wh=image_wh)
        elif (
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
