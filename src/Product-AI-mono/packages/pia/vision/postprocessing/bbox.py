from typing import List, Tuple

import cv2
import numpy as np
import torch
from shapely import Point, Polygon
from torchvision.ops import box_iou


def clip(value, min_value, max_value):
    """
    값을 지정된 범위 내로 제한한다.

    Args:
        value (float): 제한할 값.
        min_value (float): 최소값.
        max_value (float): 최대값.

    Returns:
        float: 지정된 범위 내에서 조정된 값.
    """
    return max(min_value, min(value, max_value))


def xyxy2rhombus(
    x1: int, y1: int, x2: int, y2: int,
    margin_top: int = 0,
    margin_bottom: int = 0,
    margin_left: int = 0,
    margin_right: int = 0,
    expand: float = 0.15,      # 좌우 모서리 좁힘 정도 (비율)
    head_room: float = 0.30,   # 위쪽 여유 (비율)
    ground_bias: float = 0.15  # 아래쪽 바닥 쪽으로 치우침 (비율)
):
    """
    (x1, y1, x2, y2) 박스를 기반으로 마름모 점 4개를 반환.
    마진은 '픽셀' 단위로 적용되며, 이후 비율 기반 보정이 적용됨.

    반환: [p1, p2, p3, p4] (각각 (x, y))
      - p1: 상단(머리) 쪽
      - p2: 좌측 모서리
      - p3: 하단(바닥) 쪽
      - p4: 우측 모서리
    """
    # 1) 마진 적용 (픽셀 단위)
    X1 = x1 - margin_left
    Y1 = y1 - margin_top
    X2 = x2 + margin_right
    Y2 = y2 + margin_bottom

    # 2) 중심/반폭/반높이 계산
    cx = (X1 + X2) / 2.0
    cy = (Y1 + Y2) / 2.0
    hw = (X2 - X1) / 2.0      # half width
    hh = (Y2 - Y1) / 2.0      # half height

    # 3) 마름모 네 점 계산
    #    - p1: 위쪽(머리 여유: head_room 비율만큼 아래로)
    #    - p2: 좌측(약간 안쪽으로: expand 비율만큼 x 이동)
    #    - p3: 아래쪽(바닥 쪽으로 ground_bias 비율만큼 위로)
    #    - p4: 우측(약간 안쪽으로: expand 비율만큼 x 이동)
    p1 = (int(cx), int(Y1 + hh * head_room))
    p2 = (int(X1 + hw * expand), int(cy))
    p3 = (int(cx), int(Y2 - hh * ground_bias))
    p4 = (int(X2 - hw * expand), int(cy))

    return [p1, p2, p3, p4]


def calc_intersect(bbox, rois, bbox_expand_value=0.45):  # bbox : xmin,ymin,xmax,ymax
    bbox_4_coord = xyxy2rhombus(*bbox, bbox_expand_value)

    total = 0
    poly = Polygon(rois)

    for b in bbox_4_coord:
        if poly.contains(Point(b)):
            total += 1
    if total == 4:
        return True  # 모든 점이 교차하면 해당 박스는 roi 내에 있다
    else:
        return False


def filter_small_contours(
    contours_batches: List[List[np.ndarray]],
    bbox_min_areas: List[float],
    max_bbox_num: int = 16,
) -> List[List[Tuple[int, int, int, int]]]:
    result: List[List[Tuple[int, int, int, int]]] = []
    for contours, bbox_min_area in zip(contours_batches, bbox_min_areas):
        valid: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            if len(valid) >= max_bbox_num:
                break
            if cv2.contourArea(cnt) < bbox_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            valid.append((x, y, x + w, y + h))
        result.append(valid)
    return result


def find_contours(
    masks: List[np.ndarray],
    bbox_min_areas: List[float],
    max_bbox_num: int = 16
) -> List[List[Tuple[int, int, int, int]]]:
    raw_contours = [
        cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for mask in masks
    ]
    return filter_small_contours(
        raw_contours,
        max_bbox_num=max_bbox_num,
        bbox_min_areas=bbox_min_areas
    )


def filter_bboxes_by_motion(
    color_bboxes: List[Tuple[int, int, int, int]],
    motion_mask: np.ndarray,
    min_motion_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    selects: List[Tuple[int, int, int, int]] = []
    h_mask, w_mask = motion_mask.shape
    for x1, y1, x2, y2 in color_bboxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w_mask - 1, x2), min(h_mask - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        roi = motion_mask[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            continue
        if (cv2.countNonZero(roi) / roi.size) >= min_motion_ratio:
            selects.append((x1c, y1c, x2c, y2c))
    return selects


def update_tracked_boxes(
    detections: List[Tuple[int, int, int, int]],
    tracked: List[dict],
    ttl_frames: int,
    iou_thresh: float,  # IOU_THRESH -> 인자화
) -> List[dict]:
    """
    IoU 기반의 간단한 최근접-갱신 트래킹.
    - 각 tracked에 ttl 적용/감소
    - 새 detection은 IoU 최고 매칭과 비교해 갱신 또는 신규 생성
    """
    for tb in tracked:
        tb["ttl"] -= 1
        tb["matched"] = False
    tracked = [tb for tb in tracked if tb["ttl"] > 0]

    if len(tracked) > 0:
        tb_boxes = torch.tensor([tb["bbox"] for tb in tracked], dtype=torch.float)
    else:
        tb_boxes = torch.empty((0, 4), dtype=torch.float)

    for det in detections:
        det_box = torch.tensor(det, dtype=torch.float).unsqueeze(0)
        matched = False

        if tb_boxes.numel() > 0:
            ious = box_iou(tb_boxes, det_box).squeeze(1)
            max_iou, idx = ious.max(0)
            if float(max_iou) >= iou_thresh:
                tb = tracked[int(idx)]
                tb["bbox"] = det
                tb["ttl"] = ttl_frames
                tb["matched"] = True
                matched = True
        if not matched:
            tracked.append({"bbox": det, "ttl": ttl_frames, "matched": True})
            # tb_boxes를 즉시 갱신하고 싶으면 아래 줄 활성화:
            # tb_boxes = torch.cat([tb_boxes, det_box], dim=0) if tb_boxes.numel() > 0 else det_box

    return tracked


def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    max_w: int,
    max_h: int,
    top_ratio: float = 0.,
    bottom_ratio: float = 0.,
    left_ratio: float = 0.,
    right_ratio: float = 0.
) -> Tuple[int, int, int, int]:
    w, h = x2 - x1, y2 - y1

    # 방향별 확장 크기 계산
    dt = int(h * top_ratio)
    db = int(h * bottom_ratio)
    dl = int(w * left_ratio)
    dr = int(w * right_ratio)

    # 확장된 좌표 계산 (이미지 경계를 넘지 않도록 제한)
    nx1 = max(0, x1 - dl)
    ny1 = max(0, y1 - dt)
    nx2 = min(max_w, x2 + dr)
    ny2 = min(max_h, y2 + db)

    # 유효한 bbox인지 확인
    return (nx1, ny1, nx2, ny2) if nx2 > nx1 and ny2 > ny1 else (x1, y1, x2, y2)


def expand_batch(
    batch_bboxes: List[Tuple[int, int, int, int]],
    max_w: int,
    max_h: int,
    top_ratio: float = 0.,
    bottom_ratio: float = 0.,
    left_ratio: float = 0.,
    right_ratio: float = 0.
) -> List[Tuple[int, int, int, int]]:
    return [
        expand_bbox(
            *bb,
            max_w=max_w,
            max_h=max_h,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            left_ratio=left_ratio,
            right_ratio=right_ratio
        ) for bb in batch_bboxes
    ]


def expand_batches_bboxes(
    batched_bboxes: List[List[Tuple[int, int, int, int]]],
    max_w: int,
    max_h: int,
    top_ratio: float = 0.,
    bottom_ratio: float = 0.,
    left_ratio: float = 0.,
    right_ratio: float = 0.
) -> List[List[Tuple[int, int, int, int]]]:
    return [
        expand_batch(
            batch,
            max_w=max_w,
            max_h=max_h,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio,
            left_ratio=left_ratio,
            right_ratio=right_ratio
        ) for batch in batched_bboxes
    ]


def crop_batches_bboxes(
    batched_bboxes: List[List[Tuple[int, int, int, int]]],
    batches_frame: List[np.ndarray],
) -> List[List[np.ndarray]]:
    return [
        crop_batch_bboxes(
            batch_bboxes,
            frame,
        )
        for batch_bboxes, frame in zip(batched_bboxes, batches_frame)
    ]


def crop_batch_bboxes(
    batch_bboxes: List[Tuple[int, int, int, int]],
    frame: np.ndarray
) -> List[np.ndarray]:
    cropped_bbox = []
    for bbox in batch_bboxes:
        cropped_bbox.append(frame[bbox[1] : bbox[3], bbox[0] : bbox[2]])
    return cropped_bbox


def _get_ratio_modified2origin_coordinate(xyxy, w_org, h_org, w_new, h_new, pad=None):
    """Letterbox 적용된 좌표를 원본 좌표계로 역변환"""
    x1, y1, x2, y2 = xyxy
    gain = min(w_new / w_org, h_new / h_org)

    if pad is not None:
        pad_left, pad_top, pad_right, pad_bottom = pad
    else:
        # 대칭 패딩 가정 (정수화로 인한 1px 비대칭 가능)
        pad_w = w_new - w_org * gain
        pad_h = h_new - h_org * gain
        pad_left = pad_w / 2.0
        pad_top = pad_h / 2.0

    x1 = (x1 - pad_left) / gain
    x2 = (x2 - pad_left) / gain
    y1 = (y1 - pad_top) / gain
    y2 = (y2 - pad_top) / gain

    return x1, y1, x2, y2


def _get_scale_modified2origin_coordinate(xyxy, w_org, h_org, w_new, h_new):
    """단순 스케일 적용된 좌표를 원본 좌표계로 역변환"""
    x1, y1, x2, y2 = xyxy
    x_scale = w_org / w_new
    y_scale = h_org / h_new

    x1 *= x_scale
    x2 *= x_scale
    y1 *= y_scale
    y2 *= y_scale

    return x1, y1, x2, y2


def _get_origin2ratio_modified_coordinate(xyxy, w_org, h_org, w_new, h_new, pad=None):
    """원본 좌표를 letterbox 적용된 새 좌표계로 변환"""
    x1, y1, x2, y2 = xyxy
    gain = min(w_new / w_org, h_new / h_org)

    if pad is not None:
        pad_left, pad_top, pad_right, pad_bottom = pad
    else:
        pad_w = w_new - w_org * gain
        pad_h = h_new - h_org * gain
        pad_left = pad_w / 2.0
        pad_top = pad_h / 2.0

    x1 = x1 * gain + pad_left
    x2 = x2 * gain + pad_left
    y1 = y1 * gain + pad_top
    y2 = y2 * gain + pad_top

    return x1, y1, x2, y2


def _get_origin2scale_modified_coordinate(xyxy, w_org, h_org, w_new, h_new):
    """원본 좌표를 단순 스케일된 새 좌표계로 변환 (비율 유지 X)"""
    x1, y1, x2, y2 = xyxy

    x_scale = w_new / w_org
    y_scale = h_new / h_org

    x1 *= x_scale
    x2 *= x_scale
    y1 *= y_scale
    y2 *= y_scale

    return x1, y1, x2, y2


def modified2origin_coordinate(
    xyxy,
    now_shape,          # (h_new, w_new)
    original_shape,     # (h_orig, w_orig)
    ratio: bool = True,
    pad=None,  # (pad_left, pad_top, pad_right, pad_bottom) - 선택
):
    """
    전처리(리사이즈/레터박스)된 좌표(x1,y1,x2,y2)를 원본 이미지 좌표로 변환.
    - ratio=True: letterbox 가정(비율 유지 + 패딩). pad가 주어지면 비대칭 패딩도 정확히 처리.
    - ratio=False: 단순 비율 스케일링(가로/세로 각각 다른 배율).
    """
    h_new, w_new = now_shape
    h_org, w_org = original_shape

    if not (h_new > 0 and w_new > 0 and h_org > 0 and w_org > 0):
        raise ValueError("Invalid shapes")

    if ratio:
        # letterbox 역변환
        x1, y1, x2, y2 = _get_ratio_modified2origin_coordinate(xyxy, w_org, h_org, w_new, h_new, pad)
    else:
        # 단순 스케일 역변환 (비율 유지 X)
        x1, y1, x2, y2 = _get_scale_modified2origin_coordinate(xyxy, w_org, h_org, w_new, h_new)

    # 좌표 정렬(혹시 뒤집혀 왔을 때 대비)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    # 반올림 + 클램프
    x1 = int(round(max(0, min(w_org - 1, x1))))
    y1 = int(round(max(0, min(h_org - 1, y1))))
    x2 = int(round(max(0, min(w_org - 1, x2))))
    y2 = int(round(max(0, min(h_org - 1, y2))))

    return [x1, y1, x2, y2]


def origin2modified_coordinate(
    xyxy,
    original_shape,  # (h_orig, w_orig)
    now_shape,  # (h_new,  w_new)
    ratio: bool = True,
    pad=None,   # (pad_left, pad_top, pad_right, pad_bottom)
):
    """
    원본 이미지 좌표(x1,y1,x2,y2)를 전처리(리사이즈/레터박스)된 이미지 좌표로 변환.

    Args:
        original_shape: (h_orig, w_orig)
        now_shape:      (h_new,  w_new)
        ratio: True 이면 letterbox 가정(비율 유지 + 패딩), False면 단순 스케일
        pad:   비대칭 패딩 사용 시 (left, top, right, bottom). None이면 대칭 패딩 가정

    Returns:
        (x1, y1, x2, y2)  # int, now_shape 범위로 클램프됨
    """
    h_org, w_org = original_shape
    h_new, w_new = now_shape

    if not (h_new > 0 and w_new > 0 and h_org > 0 and w_org > 0):
        raise ValueError("Invalid shapes")

    if ratio:
        # letterbox 순방향: x' = x * gain + pad_left, y' = y * gain + pad_top
        x1, y1, x2, y2 = _get_origin2ratio_modified_coordinate(xyxy, w_org, h_org, w_new, h_new, pad)
    else:
        # 단순 비율 스케일(비율 유지 X)
        x1, y1, x2, y2 = _get_origin2scale_modified_coordinate(xyxy, w_org, h_org, w_new, h_new)

    # 좌표 정렬(혹시 뒤집혀 있을 때 대비)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    # 반올림 + 클램프(now_shape 경계 내)
    x1 = int(round(max(0, min(w_new - 1, x1))))
    y1 = int(round(max(0, min(h_new - 1, y1))))
    x2 = int(round(max(0, min(w_new - 1, x2))))
    y2 = int(round(max(0, min(h_new - 1, y2))))

    return x1, y1, x2, y2
