from collections import defaultdict

import numpy as np
from pia_prod.AI.global_config import (
    ROI_POLYGON_COORDINATES_KEY,
    DIVIDE_COORDINATS_KEY,
    DIVIDED_ROI_KEY,
    ROI_KEY,
    EXPANDED_ROI_KEY,
    CAMERA_ID_KEY,
)
from typing import List
from pia.vision.roi.roi_manager import ROIManagerBase
from pia_prod.AI.modules.vehicle_reverse.config import VEHICLE_REVERSE_CV_CATEGORY


class VehicleReverseRoIManager(ROIManagerBase):
    """
    RoI(Region of Interest)을 관리하는 클래스.

    기능:
    - RoI 정보를 저장 및 관리
    - RoI 좌표 변환 및 확장 기능 제공
    - RoI를 기준으로 이미지를 crop하여 추출 가능
    - 다각형 분할 및 ROI 업데이트 기능 포함
    """

    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def find_direction_vector(line_points):
        """
        주어진 선의 방향 벡터를 계산한다.

        Args:
            line_points (list): 선의 두 점 좌표 리스트.

        Returns:
            np.array: 방향 벡터.
        """
        p1 = np.array(line_points[0])
        p2 = np.array(line_points[1])
        direction_vector = p2 - p1
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return direction_vector
        return direction_vector / norm

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        if len(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]) != 0:
            coordinate = self.get_pair_list(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY])
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]

        divide_coordinates = self.get_pair_list(roi_raw_info[ROI_KEY][DIVIDE_COORDINATS_KEY])
        direction_vector = self.find_direction_vector(divide_coordinates)

        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = roi_raw_info[ROI_KEY][
            ROI_POLYGON_COORDINATES_KEY
        ]  # 원본 RoI 기억
        self.roi_dict[camera_id][
            EXPANDED_ROI_KEY
        ] = coordinate  # 2개씩 나눠진 좌표값을 가진 것을 EXPANDED_ROI_KEY에 저장
        self.roi_dict[camera_id][DIVIDE_COORDINATS_KEY] = roi_raw_info[ROI_KEY][
            DIVIDE_COORDINATS_KEY
        ]  # 원본 방향벡터 파라미터 기억
        self.roi_dict[camera_id][DIVIDED_ROI_KEY] = direction_vector  # 실제 방향 벡터

    def get_roi_info(self, camera_id: str, roi_raw_info: dict, image_wh: tuple) -> dict:
        """
        특정 카메라의 RoI 정보를 가져온다. 필요한 경우, RoI 정보를 새롭게 등록한다.

        Args:
            camera_id (str): 카메라 ID.
            roi_type (int): RoI 타입 (0: 기본 RoI, 1: 분할된 RoI).
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
        ) or (
            roi_raw_info[ROI_KEY][DIVIDE_COORDINATS_KEY]
            != self.roi_dict[camera_id][DIVIDE_COORDINATS_KEY]
        ):
            self.add_roi(camera_id=camera_id, roi_raw_info=roi_raw_info, image_wh=image_wh)

        return self.roi_dict[camera_id]

    def get_rois_info(self, batches: List[np.array], user_params: List) -> List[dict]:
        roi_results = []
        direction_vector_results = []

        for idx, (batch, user_param) in enumerate(zip(batches, user_params)):
            cv_event = user_param["user_param"]["cvEvent"]

            found_key = None
            for key in VEHICLE_REVERSE_CV_CATEGORY:
                if key in cv_event:
                    found_key = key
                    break

            roi_dict = self.get_roi_info(
                camera_id=user_param["user_param"][CAMERA_ID_KEY],
                roi_raw_info=cv_event[found_key],
                image_wh=tuple(reversed(batch.shape[:2])),
            )

            roi_results.append(roi_dict[EXPANDED_ROI_KEY])
            direction_vector_results.append(roi_dict[DIVIDED_ROI_KEY])
        return roi_results, direction_vector_results
