import numpy as np
import cv2

from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.kumho_pinch.config import (
    PINCH_ALARM_DURATION,
    PINCH_QUEUE_SIZE,
)


# Pinch 이벤트
class PinchEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "pinch_cv"
        self.alarm_duration = PINCH_ALARM_DURATION
        self.duration_queue = defaultdict(lambda: deque(maxlen=PINCH_QUEUE_SIZE))
        self.event_status = defaultdict(int)

    def update(self, results, stream_ids, rois):
        alarms = []
        for tracks, stream_id in zip(results, stream_ids):
            intrusion_detected = 0  # 현재 프레임 침범 여부, 1은 침범

            roi_polygon = rois[stream_id].get("letterbox_roi", None)

            # ROI가 없으면 침범 판정 스킵하되, 큐에는 0을 추가
            if roi_polygon is None:
                self.duration_queue[stream_id].append(0)
                continue

            # 최적화: bbox마다 처리하지 말고 tracks가 있을 때 한번에 처리하는 방식
            if len(tracks) == 0:
                self.duration_queue[stream_id].append(0)
                continue

            # tracks가 tensor인 경우 한 번에 변환
            if hasattr(tracks, 'cpu'):
                tracks = tracks.cpu().numpy()

            # 검출된 모든 Person에 대한 침범 여부 확인
            for track in tracks:
                # 빈 track은 스킵
                if len(track) == 0:
                    continue

                # Person 바운딩 박스 좌표 추출
                bbox = track[:4]  # [x1, y1, x2, y2]

                # 이미 numpy로 변환되었으므로 추가 변환 불필요

                # 바운딩 박스 중심 기준으로 20% 축소 (가로/세로 각각 80%)
                shrinked_bbox = self._shrink_bbox(bbox, shrink_ratio=0.8)
                # 축소된 바운딩 박스의 4개 꼭짓점 좌표 추출
                corners = self._get_bbox_corners(shrinked_bbox)

                if self._check_any_point_in_polygon(corners, roi_polygon):
                    intrusion_detected = 1
                    break  # 한 명이라도 침범하면 해당 프레임은 침범

            # 현재 프레임의 침범 여부 큐에 추가
            self.duration_queue[stream_id].append(intrusion_detected)

        # 침범 횟수 체크하여 알람 발생 여부 결정
        alarms = self.check_alarm_duration()
        return alarms

    @staticmethod
    def _shrink_bbox(bbox, shrink_ratio=0.8):
        # 바운딩 박스 좌표 추출
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        # 원본 바운딩 박스의 너비와 높이 계산
        w, h = x2 - x1, y2 - y1
        # 바운딩 박스의 중심점 계산
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # 중심점을 기준으로 축소된 바운딩 박스 너비와 높이 계산
        new_w, new_h = w * shrink_ratio, h * shrink_ratio
        # 중심점 기준으로 축소된 바운딩 박스의 새로운 좌표 계산
        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def _get_bbox_corners(bbox):
        x1, y1, x2, y2 = bbox
        return [
            [x1, y1],  # 좌상단
            [x2, y1],  # 우상단
            [x2, y2],  # 우하단
            [x1, y2],  # 좌하단
        ]

    @staticmethod
    def _check_any_point_in_polygon(points, polygon):
        polygon = np.array(polygon, dtype=np.int32)
        for point in points:
            # cv2.pointPolygonTest: 내부/경계면 >= 0, 외부면 < 0
            result = cv2.pointPolygonTest(polygon, tuple(point), False)
            if result >= 0:  # 내부 또는 경계
                return True
        return False

    def check_alarm_duration(self):
        alarms = {}
        for key, value in self.duration_queue.items():
            before_status = self.event_status[key]

            # PINCH_QUEUE_SIZE 프레임 중 PINCH_ALARM_DURATION 프레임 이상 침범 확인
            is_alarm = int(sum(value) >= self.alarm_duration)

            now_status = self.STATUS_TRANSITION[before_status][is_alarm]
            self.event_status[key] = now_status

            if now_status in [1, 3]:  # 알람 상태
                alarms[key] = [self.EVENT_STATUS_DICT[now_status], None]

        return alarms
