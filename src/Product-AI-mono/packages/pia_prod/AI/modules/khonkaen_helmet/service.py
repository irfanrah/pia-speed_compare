from typing import List, Dict, Tuple
import numpy as np
import torch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.khonkaen_helmet.config import (
    DEVICE,
    MODEL_BIKER_DETECTION_TRT_PATH,
    MODEL_HELMET_CLS_TRT_PATH,
    OD_TARGET_CLASSES,
    OD_INPUT_SIZE,
    CLS_INPUT_SIZE,
    TARGET_CATEGORY_INDEX,
    CLS_CONFIDENCE_THRESHOLD,
    LIMITED_NUM_OF_CAMERA,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_NMS_THRESHOLD,
    OD_CONFIDENCE_THRESHOLD,
)

from pia_prod.AI.modules.khonkaen_helmet.event import HelmetEventManager
from pia_prod.AI.modules.khonkaen_helmet.roi_manager import HelmetRoIManager
from pia.vision.preprocessing.resize import LetterBoxTorch
from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia_prod.AI.utils.utils import threshold_check
from pia_prod.AI.modules.khonkaen_helmet.debug_utils import save_snapshot_for_odcls
from pia.vision.postprocessing.bbox import modified2origin_coordinate
from pia_prod.AI.utils.utils import free_autobackend
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class HelmetService(ServiceBase):
    ZERO_TENSOR = torch.tensor([0], device=DEVICE)
    TRUE_TENSOR = torch.tensor(True, device=DEVICE)
    CLS_CONFIDENCE_THRESHOLD_TENSOR = torch.tensor(CLS_CONFIDENCE_THRESHOLD, device=DEVICE)

    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.category_name = "helmet"
        self.save_video = False  # Debugging 용도
        self.video_manager = None  # Debugging 용도
        self.detected_counter = 0  # Debugging 용도

    def __del__(self):
        self.del_model()

    def _init_values(self):
        self.od_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA, target_size=OD_INPUT_SIZE, device=DEVICE
        )
        self.cls_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA * LIMITED_NUM_OF_PERSON_PER_CAMERA,
            target_size=CLS_INPUT_SIZE,
            device=DEVICE,
        )

    def _load_roi_manager(self):
        return HelmetRoIManager()

    def _load_model(self):
        self.object_detection_instance = PiaONNXTensorRTModel(
            model_path=MODEL_BIKER_DETECTION_TRT_PATH, device=DEVICE
        )
        self.classification_instance = PiaONNXTensorRTModel(
            model_path=MODEL_HELMET_CLS_TRT_PATH, device=DEVICE
        )

    def del_model(self):
        """모델을 삭제합니다."""
        if hasattr(self, "object_detection_instance"):
            free_autobackend(self.object_detection_instance)

        if hasattr(self, "classification_instance"):
            free_autobackend(self.classification_instance)

    def _load_event_manager(self):
        return HelmetEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        """AI inference를 수행하고 결과를 rabbitMQ에 전송합니다.

        Args:
            batches (List[np.array]): RTSP로부터 받아온 len(batches)개의 이미지
            stream_ids (List[str]): 카메라의 stream_id
            user_params (List[AddStreamModel]): AddStreamModel 데이터클래스에 정의된 파라미터들

        Returns:
            None
        """
        (cropped_images, matched_info, total_bbox_info, cropped_batches, categories_batches) = (
            self.get_pgie_results(batches=batches, stream_ids=stream_ids, user_params=user_params)
        )

        # cls는 병렬처리이므로 for문 밖에서 처리
        cls_results, raw_cls_results = self.get_classification_result(
            cropped_images, matched_info
        )  # Classification

        self.get_alarm_helmet(cls_results, matched_info)  # Event Manager, Get Alarm

        alarms = {}
        for total_idx, origin_batch_idx in enumerate(categories_batches):  # Send Alarm
            state = self.alarm_event_manager.event_state[stream_ids[origin_batch_idx]]
            if state in [0, 2]:  # 알람이 없거나 계속 진행중일 땐 메세지를 보내지 않음
                continue
            is_start = True if state == 1 else False  # 알람이 처음 생긴 경우 thumbnail을 보냄
            alarms[stream_ids[origin_batch_idx]] = [is_start, None]

        if self.logging_flag:
            save_snapshot_for_odcls(
                images=cropped_batches,
                stream_ids=stream_ids,
                bboxes=total_bbox_info,
                raw_cls_results=raw_cls_results,
                category_index=TARGET_CATEGORY_INDEX,
                classify_matched_info=matched_info,
                video_mode=self.save_video,
                video_instance=self.video_manager,
            )

        self.frame_cnt += 1
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
        batches: List[np.array],
        stream_ids: List[str],
        user_params: List,
    ) -> Tuple[List[np.array], Dict[str, int]]:
        """N개로 들어오는 이미지를 받아 object detection task를 수행하고, 결과를 반환합니다.

        Args:
            batches (List[np.array]): len(batches)개만큼 들어오는 이미지
            stream_ids (list[str]): 카메라의 stream_ids
            user_params (dict): AddStreamModel 데이터클래스에 정의된 파라미터들. ROI 정의시 사용됨

        Returns:
            crop_list(List[np.array]): detected된 object를 crop한 이미지 리스트
            matched_info(Dict[str, int]): stream_ids 당 몇개씩 detection 되었는지 카운트를 알 수 있는 변수

        NOTE:
            categories_batches
                >>> [0, 1, 0, 0]  # 0은 월담, 1은 따라들어가기 순으로 배치 들어온것임. len(batches)보다 클 수 있음
            divided_roi_info[0][2]
                >>> array([[456, 411],
                    [458, 411],
                    [633, 419],
                    [612, 205],
                    [465, 175]], dtype=int32)
            # 0번째 categories_batches의 2번째 ROI 좌표. 이안에 bbox가 검출되었는지 확인하는 용도로 쓰인다.
        """
        # 추후 nms에서 batch별로 conf, iou threshold를 다르게 주기 위해 리스트로 변경
        # conf_thres = []
        # iou_thres = []
        # for i in user_params:
        #     cv_event = i[USER_PARAM_KEY][CV_EVENT_KEY]
        #     for key in HELMET_CV_CATEGORY:
        #         if key in cv_event:
        #             conf_thres.append(cv_event[key][OD_THRESHOLD_KEY])
        #             iou_thres.append(cv_event[key][IOU_THRESHOLD_KEY])
        #             break

        # Preprocess 1 - Crop with RoI
        cropped_batches, categories_batches = self.roi_manager.process_batches_with_roi(
            batches=batches, user_params=user_params
        )

        # Preprocess 2 - Letterbox & Normalize
        resized_images = self.od_letterbox_instance(imgs=cropped_batches)
        preprocessed_images = resized_images / 255.0

        # Inference model
        raw_od_results = self.object_detection_instance(preprocessed_images)  # len() = N

        # Postprocess
        # TODO: conf_thres, iou_thres 리스트로 유저에 의해 변경될 수 있도록 변경
        nms_od_result = torch_non_max_suppression(
            prediction=raw_od_results,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=OD_TARGET_CLASSES,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )

        matched_info = {}
        cropped_images = []
        total_bbox_info = []

        for cropped_batch_idx, batch_idx_info in enumerate(categories_batches):
            cnt = 0
            now_bbox_info = []

            for pred in nms_od_result[cropped_batch_idx]:
                x1, y1, x2, y2, conf, cls = pred
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue  # 너무 작은 박스는 무시

                origin_bbox = modified2origin_coordinate(
                    xyxy=(x1, y1, x2, y2),
                    now_shape=OD_INPUT_SIZE,
                    original_shape=cropped_batches[batch_idx_info].shape[1:3],
                )
                origin_bbox.extend([conf.item(), int(cls.item())])
                now_bbox_info.append(origin_bbox)
                x1, y1, x2, y2 = origin_bbox[:4]
                cropped_image = cropped_batches[batch_idx_info][:, y1:y2, x1:x2]
                cropped_images.append(cropped_image)
                cnt += 1
            matched_info[stream_ids[batch_idx_info]] = cnt
            total_bbox_info.append(now_bbox_info)
        return cropped_images, matched_info, total_bbox_info, cropped_batches, categories_batches

    def get_classification_result(self, cropped_image: List[np.array], matched_info) -> List[bool]:
        """np.array type의 이미지를 받아 classification task를 수행하고, 결과를 반환합니다.

        Args:
            cropped_image (List[np.array]): Cropped된 이미지 리스트

        Returns:
            List[bool]: 모든 cropped_image에 대해 분류된 결과가 이상상황이면 True, 아니면 False를 반환받은 리스트
        """
        results = []
        raw_cls_results = []

        if len(cropped_image) == 0:
            return [self.ZERO_TENSOR] * len(matched_info), raw_cls_results

        resized_images = self.cls_letterbox_instance(imgs=cropped_image)
        preprocessed_images = resized_images / 255.0

        raw_cls_results = self.classification_instance(preprocessed_images)  # len() = N
        results = threshold_check(
            raw_cls_results,
            matched_info,
            TARGET_CATEGORY_INDEX,
            self.CLS_CONFIDENCE_THRESHOLD_TENSOR,
        )

        return results, raw_cls_results

    def get_alarm_helmet(self, cls_results: List[bool], matched_info: Dict[str, int]) -> None:
        """
        Classification 결과를 받아 이벤트 관리를 수행합니다.
        alarm_event_manager 클래스의 get_alarm 메소드를 호출하여 지정된 카메라들에 대해 이벤트를 관리합니다.

        Args:
            cls_results (List[bool]): Classification 결과
            matched_info (Dict[str, int]): stream_id 당 몇개씩 detection 되었는지 카운트를 알 수 있는 변수

        Returns:
            None
        """
        for idx, (camera_id, event_count) in enumerate(matched_info.items()):
            self.alarm_event_manager.update(cls_results[idx], camera_id)

        self.alarm_event_manager.get_alarm(matched_info, self.logging_flag)
