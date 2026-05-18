import numpy as np
from pia.vision.roi.bbox_manager import BoundingBoxManager


def test_update_boxes_iou_merge():
    boxes = np.array(
        [
            [0, 0, 10, 10, 0.9, 1],
            [5, 5, 15, 15, 0.8, 1],
        ],
        dtype=np.float32,
    )
    manager = BoundingBoxManager(iou_threshold=0.2, distance_threshold=0)
    merged = manager.update_boxes(boxes)

    assert merged.shape == (2, 6)  # 떨어져 있음
    assert np.allclose(merged[0][:4], [0, 0, 10, 10])


def test_update_boxes_distance_merge():
    boxes = np.array(
        [
            [0, 0, 10, 10, 0.9, 1],
            [20, 0, 30, 10, 0.8, 1],
        ],
        dtype=np.float32,
    )
    manager = BoundingBoxManager(iou_threshold=1.1, distance_threshold=30)
    merged = manager.update_boxes(boxes)

    assert merged.shape == (1, 6)
    assert np.allclose(merged[0][:4], [0, 0, 30, 10])
