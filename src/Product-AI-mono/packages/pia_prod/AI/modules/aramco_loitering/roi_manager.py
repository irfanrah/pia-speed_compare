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
from pia_prod.AI.modules.aramco_loitering.config import DEVICE

from pia_prod.AI.global_config import (
    ROI_POLYGON_COORDINATES_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
)


class LoiteringRoIManager(ROIManagerBase):
    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(dict)
        self.category_list = ["loitering_cv", "배회_cv"]

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
        self.roi_dict[camera_id][
            EXPANDED_ROI_KEY
        ] = expanded_roi  # 직사각형으로 넣자. 조각조각은 coordinate
        self.roi_dict[camera_id]['after_letterbox_calc_origin_roi'] = (
            self.find_after_letterbox_calc_origin_roi(coordinate, expanded_roi)
        )

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
        self, batches: List[np.array], stream_ids, user_params: List
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
        """
        cat_results = []
        regions = []  # 배치별 roi로 이루어진 리스트
        gpu_batches = []

        for idx, (batch, stream_id, user_param) in enumerate(zip(batches, stream_ids, user_params)):
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
            cat_results.append((idx))

        cropped_images = batch_crop_region(gpu_batches, regions)  # GPU

        return cropped_images, cat_results

    @staticmethod
    def find_after_letterbox_calc_origin_roi(origin_roi, expand_roi):
        re_calc_origin_roi = origin_roi - expand_roi[0]
        expanded_roi_shape = list(reversed(list(expand_roi[2] - expand_roi[0])))
        r, pad = calc_letterbox_parameter(source_shape=expanded_roi_shape)
        after_letterbox_calc_origin_roi = ((re_calc_origin_roi * r) + np.array(pad)).astype(
            np.int32
        )

        return after_letterbox_calc_origin_roi
