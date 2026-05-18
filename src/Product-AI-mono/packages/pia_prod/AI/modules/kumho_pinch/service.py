import torch

from collections import defaultdict
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.kumho_pinch.roi_manager import PinchRoIManager
from pia_prod.AI.modules.kumho_pinch.event import PinchEventManager
from pia_prod.AI.utils.utils import get_roi_info
from pia.ai.device import load_model_backend
from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia.vision.preprocessing.resize import LetterBoxTorch
from pia_prod.AI.modules.kumho_pinch.config import (
    DEVICE,
    PERSON_DETECTION_MODEL_TRT_PATH,
    LIMITED_NUM_OF_CAMERA,
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


class PinchService(ServiceBase):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()

    def _init_values(self):
        self.device = load_model_backend(DEVICE)
        self.frame_infos = defaultdict()
        self.od_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA,  # 예상되는 최대 배치 크기
            target_size=OD_INPUT_SIZE,  # (640, 640)
            pad_color=(114, 114, 114),
            device=DEVICE,  # "cuda"
        )

    def _load_roi_manager(self):
        return PinchRoIManager()

    def _load_model(self):
        self.person_model = PiaONNXTensorRTModel(
            PERSON_DETECTION_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        return PinchEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        # ROI 좌표 정보 저장
        rois = get_roi_info(user_params=user_params)

        # ROI 정보를 저장하고 원본 이미지 반환
        original_batches = self.roi_manager.process_batches_with_roi(
            batches=batches,
            stream_ids=stream_ids,
            rois=rois,
            target_size=OD_INPUT_SIZE,
        )

        # Numpy → Torch Tensor 변환
        torch_batches = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1) for img in original_batches]
        ).to(self.device)

        # LetterBoxTorch 적용
        preprocessed_images = self.od_letterbox_instance(torch_batches) / 255.0

        # Model Inference
        raw_person_od_result = self.person_model(preprocessed_images)

        # NMS
        person_od_result = torch_non_max_suppression(
            raw_person_od_result,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=[0],
            agnostic=True,
            pre_topk=1024,
            max_det_per_img=LIMITED_NUM_OF_CAMERA,
        )

        alarms = self.alarm_event_manager.update(
            results=person_od_result,
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
