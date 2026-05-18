from pia_prod.AI.bases.service_base import ServiceBase
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.modules.yeonsei_falldown.event import FalldownEventManager
from pia_prod.AI.utils.utils import get_roi_info
from pia.ai.device import load_model_backend
from pia.vision.postprocessing.keypoint import get_keypoint_result
from pia.vision.preprocessing.resize import resize_batches
from pia_prod.AI.modules.yeonsei_falldown.config import (
    PERSON_KEYPOINT_MODEL_TRT_PATH,
    KP_INPUT_SIZE,
    DEVICE,
)
import torch
from typing import List
from pia.vision.preprocessing.resize import LetterBox
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class FalldownService(ServiceBase):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)

    def _init_values(self):
        self.letterbox_instance = LetterBox(
            new_shape=KP_INPUT_SIZE, auto=False, scaleup=True, stride=32
        )

    def _load_event_manager(self):
        self.event_manager = FalldownEventManager()

    def _load_model(self):
        self.model = PiaONNXTensorRTModel(
            PERSON_KEYPOINT_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        rois = get_roi_info(user_params)  # ROI 정보 파싱
        roi_erased_batches, _ = self.roi_manager.erase_roi(batches, rois)
        tensor_batches, _ = resize_batches(
            roi_erased_batches, letterbox_instance=self.letterbox_instance, device=DEVICE
        )
        result: List[torch.Tensor] = self.model(tensor_batches)
        result = get_keypoint_result(result, batches, KP_INPUT_SIZE)

        alarms, _, _ = self.event_manager(batches, stream_ids, user_params, result)

        if len(alarms) > 0:
            return {
                ALARMS_KEY: alarms,
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
        else:
            return None
