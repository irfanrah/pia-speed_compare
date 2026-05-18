from collections import defaultdict
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.ax_harness.roi_manager import HarnessRoIManager
from pia_prod.AI.modules.ax_harness.event import HarnessEventManager
from pia_prod.AI.utils.utils import get_roi_info
from pia.ai.device import load_model_backend
from pia.vision.postprocessing.nms import non_max_suppression
from pia.vision.preprocessing.resize import preprocess_images
from pia.vision.preprocessing.resize import LetterBox
from pia_prod.AI.modules.ax_harness.config import (
    DEVICE,
    HARNESS_DETECTION_MODEL_TRT_PATH,
    LIMITED_NUM_OF_HARNESS_PER_CAMERA,
    OD_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
    OD_INPUT_SIZE,
)
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class HarnessService(ServiceBase):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()

    def _init_values(self):
        self.device = load_model_backend(DEVICE)
        self.frame_infos = defaultdict()
        self.od_letterbox_instance = LetterBox(
            new_shape=OD_INPUT_SIZE, scaleup=True, auto=False, stride=32
        )

    def _load_roi_manager(self):
        return HarnessRoIManager()

    def _load_model(self):
        self.harness_model = PiaONNXTensorRTModel(
            HARNESS_DETECTION_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        self.harness_event_manager = HarnessEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        # roi 좌표 정보를 저장 - 있는거는 그대로, 없는거만
        rois = get_roi_info(user_params=user_params)

        # roi를 확대하여 crop하여 반환
        cropped_batches = self.roi_manager.process_batches_with_roi(
            batches=batches, stream_ids=stream_ids, rois=rois
        )

        preprocessed_images, cropped_letter_im = preprocess_images(
            ims=cropped_batches,
            device=self.device,
            letterbox_instance=self.od_letterbox_instance,  # (N, 3, 640, 640)
        )

        # model Inference
        raw_harness_od_result = self.harness_model(preprocessed_images)

        # NMS
        harness_od_result = non_max_suppression(
            raw_harness_od_result,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            agnostic=True,
            iou_thres=OD_NMS_THRESHOLD,
            classes=[1],  # 0: 하네스, 1: 하네스 미착용
            max_det=LIMITED_NUM_OF_HARNESS_PER_CAMERA,
        )

        alarms = self.harness_event_manager(
            results=harness_od_result,
            stream_ids=stream_ids,
            rois=self.roi_manager.roi_dict,
        )

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
