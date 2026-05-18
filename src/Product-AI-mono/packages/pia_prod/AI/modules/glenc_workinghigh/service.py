from collections import defaultdict
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.glenc_workinghigh.roi_manager import WorkinghighRoIManager
from pia_prod.AI.modules.glenc_workinghigh.event import WorkinghighEventManager
from pia.ai.device import load_model_backend
from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia.vision.postprocessing.bbox import modified2origin_coordinate
from pia_prod.AI.modules.glenc_workinghigh.debug_utils import save_snapshot_for_od
from pia.vision.preprocessing.resize import LetterBoxTorch
from pia_prod.AI.modules.glenc_workinghigh.config import (
    DEVICE,
    PERSON_DETECTION_MODEL_TRT_PATH,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
    OD_INPUT_SIZE,
    TARGET_CLASSES,
)
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class WorkinghighService(ServiceBase):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.category_name = "workinghigh"
        self.save_video = False  # Debugging 용도
        self.video_manager = None  # Debugging 용도
        self.detected_counter = 0  # Debugging 용도

    def _init_values(self):
        self.device = load_model_backend(DEVICE)
        self.frame_infos = defaultdict()
        self.od_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_PERSON_PER_CAMERA, target_size=OD_INPUT_SIZE, device=DEVICE
        )

    def _load_roi_manager(self):
        return WorkinghighRoIManager()

    def _load_model(self):
        self.person_model = PiaONNXTensorRTModel(
            PERSON_DETECTION_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        self.workinghigh_event_manager = WorkinghighEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        person_od_result, total_bbox_info, frame_batches = self.get_pgie_results(
            batches=batches,
            stream_ids=stream_ids,
            user_params=user_params,
        )

        alarms = self.workinghigh_event_manager(
            results=person_od_result,
            stream_ids=stream_ids,
            rois=self.roi_manager.roi_dict,
        )

        if self.logging_flag:
            save_snapshot_for_od(
                images=frame_batches,
                stream_ids=stream_ids,
                bboxes=total_bbox_info,
                category_index=TARGET_CLASSES,
                video_mode=self.save_video,
                video_instance=self.video_manager,
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

    def get_pgie_results(
        self,
        batches,
        stream_ids,
        user_params,
    ):

        total_bbox_info = []

        # frame 을 그대로 GPU tensor 로 가져옴 (RoI 외부 mask / crop 없음)
        frame_batches = self.roi_manager.process_batches_with_roi(
            batches=batches, stream_ids=stream_ids, user_params=user_params
        )

        # Preprocess - Letterbox & Normalize (frame 전체)
        resized_images = self.od_letterbox_instance(imgs=frame_batches)
        preprocessed_images = resized_images / 255.0

        # model Inference
        raw_person_od_result = self.person_model(preprocessed_images)

        # NMS
        person_od_result = torch_non_max_suppression(
            prediction=raw_person_od_result,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=TARGET_CLASSES,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )

        # Get bbox info
        if self.logging_flag:
            for batch_idx, batch in enumerate(person_od_result):
                now_bbox_info = []
                for pred in batch:
                    x1, y1, x2, y2, conf, cls = pred
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    origin_bbox = modified2origin_coordinate(
                        xyxy=(x1, y1, x2, y2),
                        now_shape=OD_INPUT_SIZE,
                        original_shape=frame_batches[batch_idx].shape[1:3],
                    )
                    origin_bbox.extend([conf.item(), int(cls.item())])
                    now_bbox_info.append(origin_bbox)

                total_bbox_info.append(now_bbox_info)

        return person_od_result, total_bbox_info, frame_batches
