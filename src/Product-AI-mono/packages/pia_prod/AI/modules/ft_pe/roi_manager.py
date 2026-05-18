from collections import defaultdict
from typing import List

import numpy as np
import torch

from pia.vision.roi.roi_manager import ROIManagerBase, batch_crop_region
from pia_prod.AI.global_config import (
    CAMERA_ID_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
    ROI_POLYGON_COORDINATES_KEY,
)
from pia_prod.AI.modules.ft_pe.config import ALL_CATEGORIES, DEVICE


class FTPERoIManager(ROIManagerBase):
    """
    FT_PE용 ROI 매니저.

    - ALL_CATEGORIES(모든 abnormal ret_event) 중 첫 매칭 카테고리의 ROI를 사용
    - 매칭 없으면 전체 프레임을 ROI로 사용
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

        for batch, user_param in zip(batches, user_params):
            ret_event = user_param["user_param"]["retEvent"]

            found_key = None
            for key in self.roi_category_list:
                if key in ret_event:
                    found_key = key
                    break

            if found_key is None:
                roi_dict = self.get_roi_info(
                    camera_id=user_param["user_param"][CAMERA_ID_KEY],
                    roi_raw_info={ROI_KEY: {ROI_POLYGON_COORDINATES_KEY: []}},
                    image_wh=tuple(reversed(batch.shape[:2])),
                )
            else:
                roi_dict = self.get_roi_info(
                    camera_id=user_param["user_param"][CAMERA_ID_KEY],
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
