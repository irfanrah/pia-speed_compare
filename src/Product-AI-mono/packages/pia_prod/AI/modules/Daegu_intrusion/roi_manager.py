from collections import defaultdict
from typing import List
import numpy as np
from pia.vision.roi.roi_manager import (
    ROIManagerBase,
    calc_expand_coord,
    calc_letterbox_parameter,
    batch_crop_region,
)
import torch
from pia_prod.AI.modules.Daegu_intrusion.config import DEVICE

from pia_prod.AI.global_config import (
    ROI_POLYGON_COORDINATES_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
)

class IntrusionRoIManager(ROIManagerBase):
    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(dict)
        self.category_list = ["intrusion_cv", "침입_cv"]
    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        if len(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]) != 0:
            coordinate = self.get_pair_list(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY])
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]
        expanded_roi = calc_expand_coord(
            roi=coordinate,
            frame_wh=image_wh,
        )
        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = roi_raw_info[ROI_KEY][
            ROI_POLYGON_COORDINATES_KEY
        ]
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = coordinate
        self.roi_dict[camera_id]['after_letterbox_calc_origin_roi'] = self.find_after_letterbox_calc_origin_roi(coordinate, expanded_roi)

    def get_roi_info(self, camera_id: str, roi_raw_info: dict, image_wh: tuple) -> dict:
        """
        특정 카메라의 RoI 정보를 가져온다. 필요한 경우, RoI 정보를 새롭게 등록한다.

        Args:
            camera_id (str): 카메라 ID.
            roi_raw_info (dict): RoI의 원본 정보 (polygonCoordinates, divideCoordinates 포함).
            image_wh (tuple): 이미지 크기 (width, height).

        Returns:
            dict: 해당 카메라의 RoI 정보.
        """
        if camera_id not in self.roi_dict:
            self.add_roi(camera_id=camera_id, roi_raw_info=roi_raw_info, image_wh=image_wh)
        if (
            roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
            != self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY]
        ):
            self.add_roi(camera_id=camera_id, roi_raw_info=roi_raw_info, image_wh=image_wh)

        return self.roi_dict[camera_id]


    def process_batches_with_roi(
        self, batches: List[np.ndarray], stream_ids: List[str], user_params: List[dict],
    ) -> List[np.ndarray]:
        """
        사용자가 입력한 ROI에 비례해서 일정크기 이상의 부분을 확대하여 Crop하여 반환한다
        """
        regions = []
        gpu_batches = []

        for idx, (stream_id, batch, user_param) in enumerate(zip(stream_ids, batches, user_params)):
            cv_event = user_param["user_param"]["cvEvent"]

            found_key = None
            for key in self.category_list:
                if key in cv_event:
                    found_key = key
                    break

            roi_dict = self.get_roi_info(
                camera_id=stream_id,
                roi_raw_info=cv_event[found_key],
                image_wh=tuple(reversed(batch.shape[:2])),
            )

            gpu_batch = (
                torch.from_numpy(batch).to(DEVICE) if isinstance(batch, np.ndarray) else batch
            )
            gpu_batches.append(gpu_batch)
            regions.append(roi_dict[EXPANDED_ROI_KEY])
        cropped_images = batch_crop_region(frames=gpu_batches, regions=regions)

        return cropped_images

    @staticmethod
    def find_after_letterbox_calc_origin_roi(
        origin_roi, expand_roi
    ):
        re_calc_origin_roi = origin_roi - expand_roi[0]
        expanded_roi_shape = list(reversed(list(expand_roi[2] - expand_roi[0])))
        r, pad = calc_letterbox_parameter(source_shape=expanded_roi_shape)
        after_letterbox_calc_origin_roi = ((re_calc_origin_roi * r) + np.array(pad)).astype(
            np.int32
        )

        return after_letterbox_calc_origin_roi
