from collections import defaultdict
from typing import List
import numpy as np
import torch
from pia.vision.roi.roi_manager import (
    ROIManagerBase,
    _to_chw,
    calc_letterbox_parameter,
)
from pia_prod.AI.modules.glenc_workinghigh.config import DEVICE

from pia_prod.AI.global_config import (
    ROI_POLYGON_COORDINATES_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
)


class WorkinghighRoIManager(ROIManagerBase):
    """RoI 관리. Daegu_intrusion 과 달리 RoI 외부 mask / crop 을 하지 않는다.

    고소작업은 사람의 일부만 RoI 안에 들어와도 이벤트로 카운트해야 하므로,
    모델 입력에서 RoI 외부를 미리 가려버리면 (Daegu_intrusion 의 mask + crop)
    오히려 검출이 누락된다. 따라서:

    - ``process_batches_with_roi`` 는 frame 전체를 GPU tensor 로 옮겨 그대로
      반환한다 (mask / crop 없음).
    - RoI 좌표는 frame 전체 letterbox 기준으로 변환해 둔다 — ``event.update``
      의 4-corner 검사가 이 좌표계에서 수행된다.
    """

    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(dict)
        self.category_list = ["workinghigh_cv", "고소작업_cv"]

    def add_roi(self, camera_id: str, roi_raw_info: dict, image_wh: tuple):
        if len(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]) != 0:
            coordinate = self.get_pair_list(roi_raw_info[ROI_KEY][ROI_POLYGON_COORDINATES_KEY])
            coordinate = self.clip_roi(roi=coordinate, w=image_wh[0], h=image_wh[1])
        else:
            coordinate = [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]

        # mask 를 칠하지 않으므로 expand 도 불필요. frame 전체를 letterbox source
        # 로 두고 RoI 좌표를 frame 전체 letterbox 좌표계로 변환한다.
        frame_box = np.array(
            [[0, 0], [0, image_wh[1]], [image_wh[0], image_wh[1]], [image_wh[0], 0]]
        )

        self.roi_dict[camera_id][ROI_POLYGON_COORDINATES_KEY] = roi_raw_info[ROI_KEY][
            ROI_POLYGON_COORDINATES_KEY
        ]
        self.roi_dict[camera_id][EXPANDED_ROI_KEY] = coordinate
        self.roi_dict[camera_id][
            "after_letterbox_calc_origin_roi"
        ] = self.find_after_letterbox_calc_origin_roi(np.array(coordinate), frame_box)

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
        self,
        batches: List[np.ndarray],
        stream_ids: List[str],
        user_params: List[dict],
    ) -> List[torch.Tensor]:
        """
        frame 을 GPU tensor 로 옮겨 그대로 반환한다. RoI 외부 mask / crop 은
        하지 않는다 (Daegu_intrusion 과의 차이점).

        부수효과: 호출 시 각 stream 의 RoI 가 ``self.roi_dict`` 에 등록된다.
        """
        gpu_batches = []

        for stream_id, batch, user_param in zip(stream_ids, batches, user_params):
            cv_event = user_param["user_param"]["cvEvent"]

            found_key = None
            for key in self.category_list:
                if key in cv_event:
                    found_key = key
                    break

            assert found_key is not None, (
                f"cvEvent 에 {self.category_list} 중 어느 카테고리 키도 없습니다. "
                f"stream_id={stream_id}, cvEvent keys={list(cv_event.keys())}"
            )

            self.get_roi_info(
                camera_id=stream_id,
                roi_raw_info=cv_event[found_key],
                image_wh=tuple(reversed(batch.shape[:2])),
            )

            # numpy frame 은 (H, W, C) 이므로 numpy 단계에서 contiguous CHW 사본을
            # 만든 후 GPU 로 복사한다. 이러면:
            #   1. LetterBoxTorch 가 기대하는 (C, H, W) 입력이 됨
            #      (Daegu_intrusion 은 batch_crop_region 내부의 _to_chw 가 자동
            #       변환했지만, mask/crop 을 빼면서 그 부수효과가 사라짐).
            #   2. 원본 numpy array 의 storage 와 GPU tensor 의 lifecycle 이
            #      분리되어, 다음 frame 이 옛 buffer 를 overwrite 할 때 발생하는
            #      "Deallocating Tensor that still has live PyObject references"
            #      경고를 피할 수 있다.
            if isinstance(batch, np.ndarray):
                chw_np = np.ascontiguousarray(batch.transpose(2, 0, 1))
                gpu_batch = torch.from_numpy(chw_np).to(DEVICE)
            else:
                gpu_batch = _to_chw(batch)
            gpu_batches.append(gpu_batch)

        return gpu_batches

    @staticmethod
    def find_after_letterbox_calc_origin_roi(origin_roi, expand_roi):
        re_calc_origin_roi = origin_roi - expand_roi[0]
        expanded_roi_shape = list(reversed(list(expand_roi[2] - expand_roi[0])))
        r, pad = calc_letterbox_parameter(source_shape=expanded_roi_shape)
        after_letterbox_calc_origin_roi = ((re_calc_origin_roi * r) + np.array(pad)).astype(
            np.int32
        )

        return after_letterbox_calc_origin_roi
