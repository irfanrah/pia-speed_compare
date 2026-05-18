from collections import defaultdict
from typing import List, Optional, Tuple
import numpy as np
from pia.vision.roi.roi_manager import (
    ROIManagerBase,
    calc_letterbox_parameter,
)


class PinchRoIManager(ROIManagerBase):
    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(dict)

    def add_roi(self, stream_id: str, roi: np.ndarray, image_shape: tuple):
        h, w = image_shape

        # 원본 ROI 저장 (비교용), 원본을 따로 저장
        self.roi_dict[stream_id]["raw_roi"] = roi

        # ROI가 없으면 None으로 저장
        if len(roi) == 0:
            self.roi_dict[stream_id]["original_roi"] = None
        else:
            # ROI를 이미지 범위 내로 클리핑
            clipped_roi = self.clip_roi(roi, w, h)
            self.roi_dict[stream_id]["original_roi"] = np.array(clipped_roi, dtype=np.int32)

        # letterbox_roi 캐시 무효화
        if "letterbox_roi" in self.roi_dict[stream_id]:
            del self.roi_dict[stream_id]["letterbox_roi"]
        if "letterbox_shape" in self.roi_dict[stream_id]:
            del self.roi_dict[stream_id]["letterbox_shape"]

    def get_roi_info(self, stream_id: str, roi: np.ndarray, image_shape: tuple) -> dict:
        # 처음 등록하는 경우
        if stream_id not in self.roi_dict:
            self.add_roi(stream_id, roi, image_shape)
        # ROI가 변경된 경우 (원본 ROI 비교), raw_roi로 비교
        elif not self._is_same_raw_roi(self.roi_dict[stream_id].get("raw_roi"), roi):
            self.add_roi(stream_id, roi, image_shape)

        return self.roi_dict[stream_id]

    def process_batches_with_roi(
        self,
        batches: List[np.ndarray],
        stream_ids: List[str],
        rois: List[np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:

        result = []
        for i, (roi, stream_id) in enumerate(zip(rois, stream_ids)):
            image_shape = (batches[i].shape[0], batches[i].shape[1])  # (h, w)

            # ROI 정보 가져오기 (변경 감지 및 업데이트 포함)
            # 모든 로직이 get_roi_info로 위임됨
            self.get_roi_info(stream_id, roi, image_shape)

            if target_size is not None:
                self.get_or_compute_letterbox_roi(
                    stream_id=stream_id,
                    original_shape=image_shape,
                    target_size=target_size,
                )

            # 원본 이미지 그대로 사용
            result.append(batches[i])

        return result

    def _is_same_raw_roi(self, roi1, roi2):
        if roi1 is None and roi2 is None:
            return True
        if roi1 is None or roi2 is None:
            return False
        if len(roi1) != len(roi2):
            return False
        if len(roi1) == 0:
            return True

        if isinstance(roi1, np.ndarray) and isinstance(roi2, np.ndarray):
            return np.array_equal(roi1, roi2)

        return np.array_equal(np.array(roi1), np.array(roi2))

    def get_or_compute_letterbox_roi(self, stream_id, original_shape, target_size):
        # 캐시 확인
        if "letterbox_roi" in self.roi_dict[stream_id]:
            cached_shape = self.roi_dict[stream_id].get("letterbox_shape")
            if cached_shape == original_shape:
                return self.roi_dict[stream_id]["letterbox_roi"]

        # 재계산
        original_roi = self.roi_dict[stream_id].get("original_roi", None)

        if original_roi is None:
            letterbox_roi = None
        else:
            scale, (pad_left, pad_top) = calc_letterbox_parameter(
                source_shape=original_shape,  # (h, w)
                target_shape=target_size,  # (target_h, target_w)
            )

            letterbox_roi = np.round(
                np.array(original_roi) * scale + np.array([pad_left, pad_top])
            ).astype(np.int32)

        # 저장
        self.roi_dict[stream_id]["letterbox_roi"] = letterbox_roi
        self.roi_dict[stream_id]["letterbox_shape"] = original_shape

        return letterbox_roi
