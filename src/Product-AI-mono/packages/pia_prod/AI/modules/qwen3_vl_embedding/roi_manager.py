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
from pia_prod.AI.modules.qwen3_vl_embedding.config import ALL_CATEGORIES, DEVICE


class Qwen3VLERoIManager(ROIManagerBase):
    """
    Multi-category RoI Manager
    - Manages RoI information for all categories: fire, falldown, violence
    - Caches latest RoI coordinates per camera
    - Automatically updates when coordinates change
    - Uses full frame if no RoI is defined
    - Crops images for subsequent preprocessing pipeline
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
        """
        입력된 배치 이미지와 사용자 매개변수를 사용하여 ROI(Region of Interest)에 해당하는 이미지 영역을 추출한다.

        Args:
            batches (List[np.array]): 입력 이미지 리스트 (각 원소는 np.array 형태의 이미지).
            user_params (List[AddStreamModel]): 각 이미지에 대한 ROI 정보가 포함된 사용자 매개변수 리스트.

        Returns:
            Tuple[List[np.array], List[Tuple[int, int]], List[List[np.array]]]:
                - images_results (List[np.array]): ROI가 적용된 이미지 리스트.
                - cat_results (List[Tuple[int, int]]): 각 ROI의 인덱스 및 카테고리 정보.

        NOTE:
            PE 의 경우 여러가지 카테고리가 들어올 수 있으므로, 배치마다
        """
        regions = []  # 배치별 roi로 이루어진 리스트
        gpu_batches = []

        for idx, (batch, user_param) in enumerate(zip(batches, user_params)):
            ret_event = user_param[USER_PARAM_KEY][RET_EVENT_KEY]

            found_key = None
            for key in self.roi_category_list:
                if key in ret_event:
                    found_key = key
                    break

            if found_key is None:  # roi 정보가 없는 경우, 전체 이미지를 roi로 사용
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

        cropped_images = batch_crop_region(gpu_batches, regions)  # GPU

        return cropped_images