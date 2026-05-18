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
from pia_prod.AI.modules.gangnam_violence.config import VIOLENCE_CATEGORY


class VQAViolenceRoIManager(ROIManagerBase):
    """
    RoI(Region of Interest)을 관리하는 클래스.

    역할:
    - vqaEvent 기반 ROI 정보를 카메라별로 캐시하고 변경 시 갱신
    - ROI polygonCoordinates 기준으로 ROI 영역만 crop
    - ROI가 하나도 없으면 원본 batches를 그대로 반환 (fast-path)
    """

    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(lambda: defaultdict(dict))
        self.category_list = VIOLENCE_CATEGORY
        self.roi_category_list = VIOLENCE_CATEGORY

    @staticmethod
    def _has_any_roi(user_params: List[dict]) -> bool:
        """
        user_params 전체에서 ROI polygonCoordinates가 1개라도 있는지 확인한다.

        Args:
            user_params (List[dict]): {user_param: {vqaEvent: {category: {roi: ...}}}} 리스트.

        Returns:
            bool: ROI가 하나라도 있으면 True, 전부 비어 있으면 False.
        """
        for user_param in user_params:
            events = user_param.get(USER_PARAM_KEY, {}).get(VQA_EVENT_KEY, {})
            for event in events.values():
                roi = event.get(ROI_KEY, {})
                if roi.get(ROI_POLYGON_COORDINATES_KEY):
                    return True
        return False

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        """
        ROI 원본 정보를 받아 카메라별 캐시에 저장한다.

        Args:
            camera_id (str): 카메라 ID.
            roi_raw_info (dict): {roi: {polygonCoordinates: [...]}} 형태.
            image_wh (tuple): (width, height) 원본 이미지 크기.
        """
        raw_roi = roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
        if len(raw_roi) != 0:
            coordinate = self.get_pair_list(raw_roi)
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]

        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = raw_roi
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = coordinate

    def get_roi_info(self, camera_id: str, roi_raw_info: dict, image_wh: tuple) -> dict:
        """
        특정 카메라의 RoI 정보를 가져온다. 필요한 경우, RoI 정보를 새롭게 등록한다.

        Args:
            camera_id (str): 카메라 ID.
            roi_raw_info (dict): RoI의 원본 정보 (polygonCoordinates 포함).
            image_wh (tuple): 이미지 크기 (width, height).

        Returns:
            dict: 해당 카메라의 RoI 정보 (polygonCoordinates, expanded ROI 포함).
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
        self, batches: List[np.array], user_params: List
    ) -> List[np.array]:
        """
        입력된 배치 이미지와 사용자 매개변수로 ROI 영역만 crop한다.
        ROI가 전혀 없으면 원본 batches를 그대로 반환한다.

        Args:
            batches (List[np.array]): 입력 이미지 리스트.
                - 각 원소는 np.ndarray (H, W, 3), BGR, uint8을 기대함.
            user_params (List[dict]): 각 이미지의 vqaEvent ROI 정보가 포함된 파라미터 리스트.
                - user_param["user_param"]["vqaEvent"][category]["roi"]["polygonCoordinates"]
                  형태의 평면 좌표 리스트([x1, y1, x2, y2, ...])를 기대함.

        Returns:
            List[np.array]:
                - ROI가 적용된 이미지 리스트 (np.ndarray, HxWxc, BGR, uint8).
                - ROI는 다각형의 bounding box로 crop되며, ROI 바깥은 0(검정)으로 채워짐.
                - ROI가 없으면 원본 batches 그대로 반환.
        """
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

        return batch_crop_region(np_batches, regions)
