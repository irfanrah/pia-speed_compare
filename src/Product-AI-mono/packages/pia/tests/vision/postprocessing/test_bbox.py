import cv2
import numpy as np
import pytest
from pia.vision.postprocessing import (
    expand_batch,
    expand_batches_bboxes,
    expand_bbox,
    filter_bboxes_by_motion,
    find_contours,
    update_tracked_boxes,
)
from pia.vision.postprocessing.bbox import (
    modified2origin_coordinate,
    origin2modified_coordinate,
)


def test_filter_small_contours_and_find_contours():
    # 마스크 1: 큰 흰색 사각형
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(mask1, (10, 10), (30, 30), 255, -1)

    # 마스크 2: 작은 흰색 사각형 (너무 작아 필터링될 예정)
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(mask2, (5, 5), (7, 7), 255, -1)

    masks = [mask1, mask2]
    results = find_contours(masks, max_bbox_num=5, bbox_min_areas=[20, 20])

    # 첫 번째 마스크에서는 1개 bbox 검출
    assert len(results[0]) == 1
    x1, y1, x2, y2 = results[0][0]
    assert (x2 - x1) > 0 and (y2 - y1) > 0

    # 두 번째 마스크는 작은 영역이라 bbox 없음
    assert len(results[1]) == 0


def test_filter_bboxes_by_motion():
    # 모션 마스크 생성 (일부 영역에만 값 있음)
    motion_mask = np.zeros((20, 20), dtype=np.uint8)
    cv2.rectangle(motion_mask, (5, 5), (15, 15), 255, -1)

    # 후보 bbox: motion 영역과 겹치는 것, 겹치지 않는 것
    color_bboxes = [
        (4, 4, 10, 10),   # motion 영역과 많이 겹침 -> 통과
        (0, 0, 3, 3),     # motion 영역과 거의 없음 -> 걸러짐
    ]

    results = filter_bboxes_by_motion(color_bboxes, motion_mask, min_motion_ratio=0.1)

    assert (4, 4, 10, 10) in results
    assert (0, 0, 3, 3) not in results


def test_update_tracked_boxes_new_and_match():
    detections = [(0, 0, 10, 10)]
    tracked = []
    ttl_frames = 3
    iou_thresh = 0.5

    # 첫 번째 프레임: 새로운 트랙 생성
    tracked = update_tracked_boxes(detections, tracked, ttl_frames, iou_thresh)
    assert len(tracked) == 1
    assert tracked[0]["bbox"] == (0, 0, 10, 10)
    assert tracked[0]["ttl"] == 3
    assert tracked[0]["matched"]

    # 두 번째 프레임: 같은 bbox 들어오면 IoU 매칭 -> 기존 트랙 갱신
    tracked = update_tracked_boxes(detections, tracked, ttl_frames, iou_thresh)
    assert len(tracked) == 1
    assert tracked[0]["ttl"] == 3  # 다시 리셋됨
    assert tracked[0]["matched"]

    # 세 번째 프레임: IoU 매칭 안 되는 bbox 들어오면 새 트랙 추가
    new_detections = [(100, 100, 110, 110)]
    tracked = update_tracked_boxes(new_detections, tracked, ttl_frames, iou_thresh)
    assert len(tracked) == 2
    assert any(tb["bbox"] == (100, 100, 110, 110) for tb in tracked)
    assert any(tb["bbox"] == (0, 0, 10, 10) for tb in tracked)


def test_update_tracked_boxes_ttl_expiration():
    detections = []
    tracked = [{"bbox": (0, 0, 10, 10), "ttl": 1, "matched": False}]
    ttl_frames = 3
    iou_thresh = 0.5

    # ttl 1이 감소 -> 0이 되어 제거됨
    updated = update_tracked_boxes(detections, tracked, ttl_frames, iou_thresh)
    assert len(updated) == 0


def test_expand_bbox_symmetric():
    # 대칭 확장
    x1, y1, x2, y2 = 100, 100, 200, 200
    max_w, max_h = 300, 300
    result = expand_bbox(x1, y1, x2, y2, max_w, max_h, 0.1, 0.1, 0.1, 0.1)
    # 예상 확장량: 10픽셀씩 좌우, 10픽셀씩 상하
    assert result == (90, 90, 210, 210)


def test_expand_bbox_asymmetric():
    # 비대칭 확장
    x1, y1, x2, y2 = 50, 50, 150, 150
    max_w, max_h = 200, 200
    result = expand_bbox(x1, y1, x2, y2, max_w, max_h, top_ratio=0.2, bottom_ratio=0.1, left_ratio=0.05, right_ratio=0.15)
    # w=100, h=100 → top=20, bottom=10, left=5, right=15
    assert result == (45, 30, 165, 160)


def test_expand_bbox_clip_to_bounds():
    # 확장된 좌표가 이미지 크기를 넘지 않도록 처리
    result = expand_bbox(0, 0, 50, 50, max_w=100, max_h=100,
                         top_ratio=0.5, bottom_ratio=0.5, left_ratio=0.5, right_ratio=0.5)
    # w=50, h=50 → 25씩 확장 → 좌상단은 0, 우하단은 75
    assert result == (0, 0, 75, 75)


def test_expand_bbox_invalid_case():
    # 확장했을 때 박스가 유효하지 않게 되는 경우 (예외적으로)
    result = expand_bbox(100, 100, 100, 100, 200, 200, 0.1, 0.1, 0.1, 0.1)
    assert result == (100, 100, 100, 100)


def test_expand_batch_multiple():
    batch_bboxes = [(10, 10, 20, 20), (0, 0, 5, 5)]
    ratio = 0.2
    results = expand_batch(
        batch_bboxes,
        max_w=100,
        max_h=100,
        top_ratio=ratio,
        left_ratio=ratio,
        right_ratio=ratio,
        bottom_ratio=ratio
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for bb, exp in zip(batch_bboxes, results):
        assert len(exp) == 4
        assert exp[0] <= bb[0]
        assert exp[1] <= bb[1]


def test_expand_batches_bboxes_nested():
    batched_bboxes = [
        [(10, 10, 20, 20), (30, 30, 40, 40)],
        [(0, 0, 5, 5)]
    ]
    ratio = 0.1
    results = expand_batches_bboxes(
        batched_bboxes,
        max_w=100,
        max_h=100,
        top_ratio=ratio,
        left_ratio=ratio,
        right_ratio=ratio,
        bottom_ratio=ratio
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(batch, list) for batch in results)
    assert all(len(bb) == 4 for batch in results for bb in batch)


@pytest.mark.parametrize("coords, orig_shape, new_shape, ratio, pad", [
    # 단순 스케일링 (비율 유지 X)
    ((10, 20, 50, 80), (100, 200), (200, 400), False, None),
    # Letterbox (대칭 padding)
    ((15, 25, 60, 100), (100, 200), (150, 300), True, None),
    # Letterbox (비대칭 padding)
    ((5, 5, 20, 20), (50, 50), (60, 80), True, (10, 5, 10, 5)),
])
def test_forward_and_inverse(coords, orig_shape, new_shape, ratio, pad):
    """원본 -> 전처리 -> 역변환 시 좌표가 일관성 있는지 확인"""

    # 원본 -> 전처리
    px1, py1, px2, py2 = origin2modified_coordinate(
        coords,
        original_shape=orig_shape,
        now_shape=new_shape,
        ratio=ratio,
        pad=pad,
    )

    # 다시 역변환
    ox1, oy1, ox2, oy2 = modified2origin_coordinate(
        (px1, py1, px2, py2),
        now_shape=new_shape,
        original_shape=orig_shape,
        ratio=ratio,
        pad=pad,
    )

    # round 오차 범위 내에서 동일해야 함
    assert (ox1, oy1, ox2, oy2) == pytest.approx(coords, abs=1)


def test_invalid_shapes():
    """음수 또는 0 크기 shape 입력 시 ValueError 발생"""
    with pytest.raises(ValueError):
        origin2modified_coordinate((0, 0, 10, 10), (0, 100), (200, 200))
    with pytest.raises(ValueError):
        modified2origin_coordinate((0, 0, 10, 10), (200, 200), (0, 100))
