from typing import List, Dict, Tuple
import numpy as np
import torch
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.bases.service_base import ServiceBase
from pia.ai.device import load_model_backend
from pia_prod.AI.modules.yonsei_walljump.config import (
    DEVICE,
    PERSON_DETECTION_MODEL_TRT_PATH,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    MODEL_WALLJUMP_CLS_TRT_PATH,
    OD_TARGET_CLASSES,
    OD_INPUT_SIZE,
    CLS_INPUT_SIZE,
    TARGET_CATEGORY_INDEX,
)
from pia_prod.AI.global_config import (
    USER_PARAM_KEY,
    CV_EVENT_KEY,
    OD_THRESHOLD_KEY,
    CLS_THRESHOLD_KEY,
    IOU_THRESHOLD_KEY,
    CAMERA_ID_KEY,
    ORGANIZATION_KEY,
)

from pia_prod.AI.modules.yonsei_walljump.event import WalljumpEventManager
from pia_prod.AI.modules.yonsei_walljump.roi_manager import WalljumpRoIManager
from pia_prod.AI.modules.yonsei_walljump.debug_utils import save_snapshot_for_odcls
from pia.vision.preprocessing.resize import preprocess_images
from pia.vision.preprocessing.resize import LetterBox
from pia_prod.AI.utils.utils import threshold_check
from pia.vision.postprocessing.nms import non_max_suppression
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class WalljumpService(ServiceBase):
    ZERO_TENSOR = torch.tensor([0], device=DEVICE)
    TRUE_TENSOR = torch.tensor(True, device=DEVICE)

    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.category_name = "walljump"

    def _init_values(self):
        self.od_letterbox_instance = LetterBox(
            new_shape=OD_INPUT_SIZE, scaleup=True, auto=False, stride=32
        )
        self.cls_letterbox_instance = LetterBox(
            new_shape=CLS_INPUT_SIZE, scaleup=True, auto=False, stride=32
        )

    def _load_roi_manager(self):
        return WalljumpRoIManager()

    def _load_model(self):
        self.object_detection_instance = PiaONNXTensorRTModel(
            PERSON_DETECTION_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )
        self.classification_instance = PiaONNXTensorRTModel(
            MODEL_WALLJUMP_CLS_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        return WalljumpEventManager()

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
        (
            cropped_images_for_walljump,
            classify_matched_info,
            raw_bbox_info_for_tailgate,
            category_info,
            divided_roi_info,
            letter_im,
        ) = self.get_object_detection_results(
            batches, stream_ids, user_params
        )  # Preprocess, Object Detection

        # cls는 병렬처리이므로 for문 밖에서 처리

        cls_results, raw_cls_results = self.get_classification_result(
            cropped_images_for_walljump, classify_matched_info, user_params
        )  # Classification
        self.get_alarm_walljump(cls_results, classify_matched_info)  # Event Manager, Get Alarm

        if self.logging_flag:
            save_snapshot_for_odcls(
                images=letter_im,
                stream_ids=stream_ids,
                bboxes=raw_bbox_info_for_tailgate,
                raw_cls_results=raw_cls_results,
                category_index=TARGET_CATEGORY_INDEX,
                classify_matched_info=classify_matched_info,
                category_name=self.category_name,
            )

        alarms = {}
        for total_idx, origin_batch_idx in enumerate(category_info):  # Send Alarm
            # user_param = user_params[origin_batch_idx]

            state = self.alarm_event_manager.event_state[stream_ids[origin_batch_idx]]
            if state in [0, 2]:  # 알람이 없거나 계속 진행중일 땐 메세지를 보내지 않음
                continue
            is_start = True if state == 1 else False  # 알람이 처음 생긴 경우 thumbnail을 보냄

            alarms[stream_ids[origin_batch_idx]] = [is_start, None]

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

    def get_object_detection_results(
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
        cropped_batches, categories_batches, divided_roi_info = (
            self.roi_manager.process_batches_with_roi(batches=batches, user_params=user_params)
        )  # Roi 가 여러개 들어오면 batch가 늘어날 수 있음
        preprocessed_images, letter_im = preprocess_images(
            ims=cropped_batches,
            device=load_model_backend(DEVICE),
            letterbox_instance=self.od_letterbox_instance,
        )  # (N, 3, 640, 640)
        raw_od_result = self.object_detection_instance(preprocessed_images)  # len() = N

        conf_thres = [
            i[USER_PARAM_KEY][CV_EVENT_KEY]['walljump_cv'][OD_THRESHOLD_KEY] for i in user_params
        ]
        iou_thres = [
            i[USER_PARAM_KEY][CV_EVENT_KEY]['walljump_cv'][IOU_THRESHOLD_KEY] for i in user_params
        ]

        nms_od_result = non_max_suppression(
            raw_od_result,
            conf_thres=conf_thres,
            agnostic=True,
            iou_thres=iou_thres,
            classes=OD_TARGET_CLASSES,
            max_det=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )  # 각 batch별 최대 인원을 7명으로 제한
        raw_bbox_info_for_walljump = []
        matched_info = {}
        crop_list_for_walljump = list()

        for cropped_batch_idx, batch_idx_info in enumerate(categories_batches):
            cnt = 0
            now_bbox_info = []
            for pred in nms_od_result[cropped_batch_idx]:
                x1, y1, x2, y2, conf, cls = map(int, pred)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(OD_INPUT_SIZE[0], x2), min(OD_INPUT_SIZE[1], y2)
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                now_bbox_info.append([x1, y1, x2, y2, conf, cls])
                cnt += 1  # 카운트 증가
                cropped_image = letter_im[cropped_batch_idx][y1:y2, x1:x2]
                crop_list_for_walljump.append(cropped_image)
            matched_info[stream_ids[batch_idx_info]] = (
                cnt  # 하나의 들어오는 이미지는 월담에 대해서 하나의 정보만 가진다
            )
            raw_bbox_info_for_walljump.append(now_bbox_info)
        return (
            crop_list_for_walljump,
            matched_info,
            raw_bbox_info_for_walljump,
            categories_batches,
            divided_roi_info,
            letter_im,
        )

    def _track_person(self, preds, stream_ids, matched_info):
        result = []
        for matched_info_each in matched_info:
            matched_idx, num_batch, category, stream_id = matched_info_each
            if category in ["intrusion", "loitering"]:
                result.append(self.tracker[stream_id][category].update(preds[matched_idx]))
        return result

    def get_classification_result(
        self, cropped_image: List[np.array], matched_info, user_params
    ) -> List[bool]:
        """np.array type의 이미지를 받아 classification task를 수행하고, 결과를 반환합니다.

        Args:
            cropped_image (List[np.array]): Cropped된 이미지 리스트

        Returns:
            List[bool]: 모든 cropped_image에 대해 분류된 결과가 이상상황이면 True, 아니면 False를 반환받은 리스트
        """
        results = []
        if len(cropped_image) == 0:
            return [self.ZERO_TENSOR] * len(matched_info), None

        cls_conf_thres = self.get_classify_threshold(user_params, matched_info)

        preprocessed_images, letter_im = preprocess_images(
            ims=cropped_image,
            device=load_model_backend(DEVICE),
            letterbox_instance=self.cls_letterbox_instance,
        )  # letterbox를 씌운 (N, 3, 224, 224) 크기 텐서
        raw_cls_results = self.classification_instance(preprocessed_images)  # len() = N
        results = threshold_check(
            raw_cls_results, matched_info, TARGET_CATEGORY_INDEX, cls_conf_thres
        )

        return results, raw_cls_results

    def get_alarm_walljump(self, cls_results: List[bool], matched_info: Dict[str, int]) -> None:
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

    def get_classify_threshold(self, user_params: List[Dict[str, any]], matched_info):
        cls_threshold_list = []
        for user_param in user_params:
            infer_key = self.make_inference_key(
                camera_id=user_param[USER_PARAM_KEY][CAMERA_ID_KEY],
                organization=user_param[USER_PARAM_KEY][ORGANIZATION_KEY],
            )
            if infer_key in matched_info:
                for _ in range(matched_info[infer_key]):
                    cls_threshold_list.append(
                        user_param[USER_PARAM_KEY][CV_EVENT_KEY]['walljump_cv'][CLS_THRESHOLD_KEY]
                    )

        return cls_threshold_list
