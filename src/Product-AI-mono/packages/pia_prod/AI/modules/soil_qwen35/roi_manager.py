from collections import defaultdict
from typing import List

import numpy as np
from pia.vision.roi.roi_manager import ROIManagerBase, batch_crop_region
from pia_prod.AI.global_config import (
    CAMERA_ID_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
    ROI_POLYGON_COORDINATES_KEY,
    USER_PARAM_KEY,
    VQA_EVENT_KEY,
)
from pia_prod.AI.modules.soil_qwen35.config import SUPPORT_CATEGORIES


class SoilQwen35RoIManager(ROIManagerBase):
    """
    Soil Qwen3.5 VQA용 RoI 관리 클래스.

    역할:
    - vqaEvent 기반 ROI 정보를 카메라별로 캐시하고 변경 시 갱신
    - ROI polygonCoordinates 기준으로 ROI 영역만 crop
    - ROI가 하나도 없으면 원본 batches를 그대로 반환 (fast-path)
    """

    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(lambda: defaultdict(dict))
        self.category_list = SUPPORT_CATEGORIES
        self.roi_category_list = SUPPORT_CATEGORIES

    @staticmethod
    def _has_any_roi(user_params: List[dict]) -> bool:
        for user_param in user_params:
            events = user_param.get(USER_PARAM_KEY, {}).get(VQA_EVENT_KEY, {})
            for event in events.values():
                roi = event.get(ROI_KEY, {})
                if roi.get(ROI_POLYGON_COORDINATES_KEY):
                    return True
        return False

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        raw_roi = roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
        if len(raw_roi) != 0:
            coordinate = self.get_pair_list(raw_roi)
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]
        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = raw_roi
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = coordinate

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
        if not self._has_any_roi(user_params):
            return batches

        regions = []
        np_batches = []

        for batch, user_param in zip(batches, user_params):
            events = user_param.get(USER_PARAM_KEY, {}).get(VQA_EVENT_KEY, {})

            found_key = None
            for key in self.roi_category_list:
                if key in events:
                    found_key = key
                    break

            if found_key is None:
                np_batches.append(batch)
                regions.append([])
                continue

            roi_raw_info = events.get(found_key, {})
            roi_coords = roi_raw_info.get(ROI_KEY, {}).get(ROI_POLYGON_COORDINATES_KEY, [])
            if not roi_coords:
                np_batches.append(batch)
                regions.append([])
                continue

            roi_dict = self.get_roi_info(
                camera_id=user_param.get(USER_PARAM_KEY, {}).get(CAMERA_ID_KEY),
                roi_raw_info=roi_raw_info,
                image_wh=tuple(reversed(batch.shape[:2])),
            )
            np_batches.append(batch)
            regions.append(roi_dict[EXPANDED_ROI_KEY])

        cropped_images = batch_crop_region(np_batches, regions)
        return cropped_images
