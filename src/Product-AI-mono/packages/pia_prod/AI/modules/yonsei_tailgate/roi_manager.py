from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import split as opsplit
from pia_prod.AI.global_config import (
    ROI_POLYGON_COORDINATES_KEY,
    DIVIDE_COORDINATS_KEY,
    DIVIDED_ROI_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
    CAMERA_ID_KEY,
)
from typing import List
from pia.vision.roi.roi_manager import ROIManagerBase, calc_letterbox_parameter
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import calc_expand_coord
from pia.vision.roi.roi_manager import cv_crop_region
from pia_prod.AI.modules.yonsei_tailgate.config import TAILGATE_CV_CATEGORY


class TailgateRoIManager(ROIManagerBase):
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
    def expand_line(line_points, origin_shape=(1920, 1080)):
        """
        주어진 선을 화면 경계까지 확장한다.

        Args:
            line_points (list): [[x1, y1], [x2, y2], ..., [xn, yn]] 형태의 좌표 리스트.
            origin_shape (tuple): 원본 이미지 크기 (width, height).

        Returns:
            list: 확장된 선 좌표 리스트.
        """
        width, height = origin_shape
        expanded_lines = []
        pair_point = [line_points[i : i + 2] for i in range(0, len(line_points), 2)]

        for line in pair_point:
            (x1, y1), (x2, y2) = line

            if x1 == x2:  # 수직선인 경우
                expanded_lines.append([(x1, 0), (x1, height)])
                continue

            # 기울기와 y절편 계산
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # 화면 경계에서의 교차점 계산
            points = []
            for x_boundary in [0, width]:
                y = slope * x_boundary + intercept
                if 0 <= y <= height:
                    points.append((x_boundary, y))
            for y_boundary in [0, height]:
                x = (y_boundary - intercept) / slope
                if 0 <= x <= width:
                    points.append((x, y_boundary))

            # 두 개의 교차점을 선택하여 확장된 선으로 추가
            if len(points) >= 2:
                expanded_lines.append([points[0], points[1]])

        return expanded_lines

    @staticmethod
    def find_divided_polygons(polygon_coords, cross_lines):
        """
        다각형을 주어진 선들에 의해 분할한다.

        Args:
            polygon_coords (list): 다각형 꼭짓점 리스트.
            cross_lines (list): 분할할 선들의 리스트.

        Returns:
            list: 분할된 다각형 좌표 리스트.
        """
        results = []
        multipolygon = MultiPolygon([Polygon(polygon_coords)])
        multiline = MultiLineString([LineString(line) for line in cross_lines])

        for line in multiline.geoms:
            multipolygon = MultiPolygon(opsplit(multipolygon, line).geoms)

        for poly in multipolygon.geoms:
            results.append(list(poly.exterior.coords)[:-1])

        return results

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        if len(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]) != 0:
            coordinate = self.get_pair_list(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY])
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]
        after_letterbox_calc_origin_roi_list = []

        divide_coordinates = self.get_pair_list(roi_raw_info[ROI_KEY][DIVIDE_COORDINATS_KEY])
        expanded_line = self.expand_line(divide_coordinates, origin_shape=image_wh)
        divided_polygons = self.find_divided_polygons(coordinate, expanded_line)
        expanded_roi = calc_expand_coord(
            roi=coordinate,
            frame_wh=image_wh,
            expand_ratio=0.0,
            horizontal_px=100,
            vertical_px=250,
        )
        for divided_polygon in divided_polygons:
            re_calc_origin_roi = divided_polygon - expanded_roi[0]
            expanded_roi_shape = list(reversed(list(expanded_roi[2] - expanded_roi[0])))
            r, pad = calc_letterbox_parameter(source_shape=expanded_roi_shape)
            after_letterbox_calc_origin_roi = ((re_calc_origin_roi * r) + np.array(pad)).astype(
                np.int32
            )
            after_letterbox_calc_origin_roi_list.append(after_letterbox_calc_origin_roi)

        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = roi_raw_info[ROI_KEY][
            ROI_POLYGON_COORDINATES_KEY
        ]
        self.roi_dict[camera_id][DIVIDE_COORDINATS_KEY] = roi_raw_info[ROI_KEY][
            DIVIDE_COORDINATS_KEY
        ]
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = expanded_roi
        self.roi_dict[camera_id][DIVIDED_ROI_KEY] = after_letterbox_calc_origin_roi_list

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
                - divided_roi_info (List[List[np.array]]): ROI 내에서 분할된 영역 정보 리스트.
        """
        images_results = []
        cat_results = []
        divided_roi_info = []

        for idx, (batch, user_param) in enumerate(zip(batches, user_params)):
            cv_event = user_param["user_param"]["cvEvent"]

            found_key = None
            for key in TAILGATE_CV_CATEGORY:
                if key in cv_event:
                    found_key = key
                    break

            roi_dict = self.get_roi_info(
                camera_id=user_param['user_param'][CAMERA_ID_KEY],
                roi_raw_info=cv_event[found_key],
                image_wh=tuple(reversed(batch.shape[:2])),
            )

            cropped_image = cv_crop_region(batch, roi_dict[EXPANDED_ROI_KEY])
            cat_results.append((idx))
            images_results.append(cropped_image)
            divided_roi_info.append(roi_dict[DIVIDED_ROI_KEY])
        return images_results, cat_results, divided_roi_info
