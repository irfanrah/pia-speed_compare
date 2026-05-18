from collections import defaultdict, deque
from typing import Dict
import numpy as np
from pia_prod.AI.modules.yeonsei_falldown.config import (
    ALARM_DURATION,
    FALL_QUEUE_SIZE,
    ANGLE_THRESHOLD,
    BBOX_CUTTING_MARGIN_RATIO,
    BBOX_KEEP_HEAD_FOOT_RATIO,
)
from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.global_config import (
    USER_PARAM_KEY,
    CV_EVENT_KEY,
)


class FalldownEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "falldown_cv"
        self.event_dict = defaultdict(lambda: deque(maxlen=FALL_QUEUE_SIZE))
        # 0 = 정상 -> 정상, 1 = 정상->알람(start), 2 = 알람->알람, 3 = 알람->정상(end)
        self.event_states = defaultdict(int)

    def update(self, batches, stream_ids, user_params, results) -> Dict[str, int]:
        head_points = []
        foot_points = []
        for i, (frame, stream_id, user_param, result) in enumerate(
            zip(batches, stream_ids, user_params, results)
        ):
            # TODO : user param 형식 변경 예정임 - 추후 수정 필요
            category_name = list(user_param[USER_PARAM_KEY][CV_EVENT_KEY].keys())[0]
            margin = user_param[USER_PARAM_KEY][CV_EVENT_KEY][category_name][
                "bbox_cutting_margin_ratio"
            ]
            min_ratio = user_param[USER_PARAM_KEY][CV_EVENT_KEY][category_name][
                "bbox_keep_head_foot_ratio"
            ]
            angle_threshold = user_param[USER_PARAM_KEY][CV_EVENT_KEY][category_name][
                "angle_threshold"
            ]
            ####################################################################
            isFalldown = False
            if len(result):
                keypoints = result[0]
                boxes = result[1]
                rs = []

                boxes, keypoints, _ = filter_edge_boxes(
                    boxes, keypoints, frame.shape, margin=margin, min_ratio=min_ratio
                )
                for bbox, point in zip(boxes, keypoints):
                    isFalldown_per_point, head_point, foot_point = self.check_falldown_condition(
                        frame, bbox, point, angle_threshold
                    )
                    rs.append(isFalldown_per_point)
                    head_points.append(head_point)
                    foot_points.append(foot_point)
                if sum(rs) > 0:
                    isFalldown = 1
                self.event_dict[stream_id].append(isFalldown)
            else:
                head_points, foot_points = [], []
                self.event_dict[stream_id].append(isFalldown)
        return self.check_alarm_duration(self.event_dict), head_points, foot_points

    def check_alarm_duration(self, event_dict: defaultdict):
        alarms = {}
        for key, value in event_dict.items():
            now_status = sum(value) >= ALARM_DURATION
            before_status = self.event_states[key]

            self.event_states[key] = self.STATUS_TRANSITION[before_status][now_status]
            if self.event_states[key] in [1, 3]:  # start or alarm
                alarms[key] = [self.EVENT_STATUS_DICT[self.event_states[key]], None]

        return alarms

    @staticmethod
    def check_falldown_condition(frame, bbox, point, angle_threshold=ANGLE_THRESHOLD):
        isFalldown = False
        frame_height, frame_width = frame.shape[:2]
        point_box = get_keypoint_bbox(point)
        foot_point = get_foot_point(point)
        head_point = get_head_point(point)

        if len(head_point) != 0 and len(foot_point) != 0:
            cond3 = check_bbox_near_edge(
                point_box, frame_width, frame_height, margin=BBOX_CUTTING_MARGIN_RATIO
            )
            if cond3:
                cond4, _ = check_head_foot_point_width(
                    head_point, foot_point, frame_width, min_ratio=BBOX_KEEP_HEAD_FOOT_RATIO
                )
                if cond4:
                    isFalldown = True
            else:
                cond5 = check_head_foot_point_coordinate(head_point, foot_point)
                if not cond5:
                    # cond6. head와 foot 좌표를 연결한 몸 중심선과 지면 간의 각도를 계산하고, 값이 임계값 이하라면 - 쓰러짐
                    cond6 = check_head_foot_point_angle(head_point, foot_point, angle_threshold)
                    if cond6:
                        isFalldown = True
                else:
                    isFalldown = True
        return isFalldown, head_point, foot_point


def check_head_foot_point_width(
    head_point, foot_point, frame_width, min_ratio=BBOX_KEEP_HEAD_FOOT_RATIO
):
    """
    머리-발 간의 가로 거리 비율이 최소 임계값 이상인지 확인
    """
    dist = abs(head_point[0] - foot_point[0])

    if frame_width != 0:
        dist_ratio = dist / frame_width
    else:
        dist_ratio = -1

    return (dist_ratio >= min_ratio), dist_ratio


def get_width_height(bbox):
    # x_min, y_min, x_max, y_max = bbox.xyxy.tolist()[0]  # [x1, y1, x2, y2] 형식
    width = bbox[2]
    height = bbox[3]
    return width, height


def bbox_aspect_ratio(bbox):
    width, height = get_width_height(bbox)

    if height != 0 and width != 0:  # width가 0이면 어짜피 0이지만 명시적으로 예외처리
        return width / height
    else:
        return 0


def check_bbox_size(bbox):
    w, h = get_width_height(bbox)

    if w <= 12 or h <= 12:
        return False
    else:
        return True


def check_bbox_near_edge(point_box, frame_width, frame_height, margin=BBOX_CUTTING_MARGIN_RATIO):
    """
    바운딩 박스 중심이 프레임 가장자리 근처에 있는지 확인
    """
    if margin < 1:
        margin_x = margin * frame_width
        margin_y = margin * frame_height
    else:
        margin_x = margin
        margin_y = margin

    center_x = (point_box[0] + point_box[2]) / 2
    center_y = (point_box[1] + point_box[3]) / 2

    return (
        (center_x <= margin_x)
        or (center_y <= margin_y)
        or (center_x >= frame_width - margin_x)
        or (center_y >= frame_height - margin_y)
    )


def check_bbox_aspect_ratio(ratio):
    if ratio <= 0.2 or ratio >= 5.0:
        return False
    else:
        return True


def get_keypoint_bbox_length(point_box):
    """
    바운딩 박스의 너비와 높이 계산
    """
    width = point_box[2] - point_box[0]
    height = point_box[3] - point_box[1]

    return width, height


def check_head_foot_point_coordinate(head, foot):
    if head[1] > foot[1]:
        return True
    else:
        return False


def get_keypoint_bbox(point):
    """
    0이 아닌 keypoint들로부터 최소 bounding box 좌표(xmin, ymin, xmax, ymax) 계산
    """
    point = np.array([p for p in point if not (p[0] == 0 and p[1] == 0)])

    x_coords = point[:, 0]
    y_coords = point[:, 1]

    x_min = np.min(x_coords)
    y_min = np.min(y_coords)
    x_max = np.max(x_coords)
    y_max = np.max(y_coords)

    return (x_min, y_min, x_max, y_max)


def check_head_foot_point_angle(head, foot, angle_threshold=ANGLE_THRESHOLD):
    dx = foot[0] - head[0]
    dy = foot[1] - head[1]
    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
    if angle <= angle_threshold:
        return True
    else:
        return False


def get_head_point(point):
    head_candidates = [
        point[0],
        point[1],
        point[2],
        point[3],
        point[4],
    ]  # nose, left_eye, right_eye, left_ear, right_ear
    head_points = [p for p in head_candidates if not np.all(p == 0)]
    if len(head_points) == 0:
        return []
    else:
        avg_point = np.mean(head_points, axis=0).astype(int)
        return avg_point


def get_foot_point(point):
    foot_candidates = [
        point[13],
        point[14],
        point[15],
        point[16],
    ]  # left_knee, right_knee, left_ankle, right_ankle
    foot_points = [p for p in foot_candidates if not np.all(p == 0)]

    ankle_left = point[15] if not np.all(point[15] == 0) else []
    ankle_right = point[16] if not np.all(point[16] == 0) else []

    if len(foot_points) == 0:
        return []

    if len(ankle_left) == 0 and len(ankle_right) == 0:
        # 발 좌표가 둘다 없으면 다른 좌표값의 평균을 사용
        avg_point = np.mean(foot_points, axis=0).astype(int)
        return avg_point
    elif len(ankle_right) == 0:  # TODO : 코드를 더 깔끔하게 쓸 수 있는 방법이 있을듯
        return ankle_left.astype(int)
    elif len(ankle_left) == 0:
        return ankle_right.astype(int)
    else:
        return np.mean([ankle_left, ankle_right], axis=0).astype(int)


def filter_edge_boxes(
    boxes: np.ndarray,
    points: np.ndarray,
    img_shape,
    margin=BBOX_CUTTING_MARGIN_RATIO,
    min_ratio=BBOX_KEEP_HEAD_FOOT_RATIO,
):
    """
    1) 박스 꼭짓점(x1,y1,x2,y2)이 'margin' 안쪽이면 남긴다
    2) 머리(0–4) ↔ 발(13–16) 평균 좌표 간 거리 ≥ min_ratio × 프레임 폭(W)이면 남긴다
       ─> 두 조건 중 하나라도 만족 → keep

    Parameters
    ----------
    boxes   : (N, ≥4) np.ndarray  |  [[x1,y1,x2,y2,conf,cls,...], ...]
    points  : (N, 17, 2) np.ndarray  |  COCO keypoints (x, y)
    img_shape : (H, W)  == frame.shape[:2]
    margin  : float(<1) → 비율  |  int(>=1) → 픽셀
    min_ratio : float  머리–발 거리 / 프레임 폭 기준 하한

    Returns
    -------
    keep_boxes  : boxes[mask]
    keep_points : points[mask]
    keep_mask   : (N,) bool np.ndarray
    """
    H, W = img_shape[:2]

    # margin 픽셀값
    m = margin * W if isinstance(margin, float) and margin < 1 else float(margin)

    b = np.asarray(boxes)
    kp = np.asarray(points)

    # ① 가장자리 기준
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    edge_mask = (x1 > m) & (y1 > m) & (x2 < W - m) & (y2 < H - m)

    # ② 머리–발 x-거리 기준 ------------------------------------------
    head = kp[:, 0:5, :]  # (N,5,2)
    feet = kp[:, 13:17, :]  # (N,4,2)

    hv = head.sum(axis=-1) != 0  # valid mask
    fv = feet.sum(axis=-1) != 0

    head_ctr_x = (head[..., 0] * hv).sum(axis=1) / np.maximum(hv.sum(axis=1), 1)
    foot_ctr_x = (feet[..., 0] * fv).sum(axis=1) / np.maximum(fv.sum(axis=1), 1)

    dist_x = np.abs(head_ctr_x - foot_ctr_x)  # ← 가로 거리
    size_mask = dist_x >= (min_ratio * W)

    keep_mask = edge_mask | size_mask
    return b[keep_mask], kp[keep_mask], keep_mask


def step1_falldown_condition(frame, bbox, margin=0.1):
    # frame_width, frame_height = frame.shape[1], frame.shape[0]
    # x_min, y_min, x_max, y_max = bbox.xyxy.tolist()[0]  # [x1, y1, x2, y2] 형식
    # w, h = get_width_height(bbox)
    # 2025 05 14 - 박스의 종횡비만을 고려하는 것은 오탐의 확률이 클 것으로 보임 - 삭제
    # if w > h:
    #     # 객체가 frame 끝쪽에 위치하는지
    #     if (x_min <= margin or x_max >= frame_width - margin or
    #             y_min <= margin or y_max >= frame_height - margin):
    #         return False
    #     else:
    #         return True
    return False
