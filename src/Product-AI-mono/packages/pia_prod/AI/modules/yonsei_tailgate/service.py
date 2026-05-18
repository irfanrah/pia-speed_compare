from typing import List, Dict, Tuple
import numpy as np
import torch
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.bases.service_base import ServiceBase
from pia.ai.device import load_model_backend
from pia_prod.AI.modules.yonsei_tailgate.config import (
    DEVICE,
    PERSON_DETECTION_MODEL_TRT_PATH,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_TARGET_CLASSES,
    OD_INPUT_SIZE,
)

from pia_prod.AI.modules.yonsei_tailgate.event import TailgateEventManager
from pia_prod.AI.modules.yonsei_tailgate.roi_manager import TailgateRoIManager
from pia_prod.AI.modules.yonsei_tailgate.func import (
    xyxy2rhombus_for_topview,
    calc_intersect_for_topview,
)
from pia.vision.preprocessing.resize import preprocess_images
from pia.vision.postprocessing.nms import non_max_suppression
from pia_prod.AI.global_config import (
    USER_PARAM_KEY,
    CV_EVENT_KEY,
    OD_THRESHOLD_KEY,
    IOU_THRESHOLD_KEY,
)
from pia.vision.preprocessing.resize import LetterBox
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class TailgateService(ServiceBase):
    ZERO_TENSOR = torch.tensor([0], device=DEVICE)
    TRUE_TENSOR = torch.tensor(True, device=DEVICE)

    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.category_name = "tailgate"

    def _init_values(self):
        self.od_letterbox_instance = LetterBox(
            new_shape=OD_INPUT_SIZE, scaleup=True, auto=False, stride=32
        )

    def _load_roi_manager(self):
        return TailgateRoIManager()

    def _load_model(self):
        self.object_detection_instance = PiaONNXTensorRTModel(
            PERSON_DETECTION_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        return TailgateEventManager()

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
            cropped_images_for_tailgate,
            classify_matched_info,
            raw_bbox_info_for_tailgate,
            category_info,
            divided_roi_info,
            letter_im,
        ) = self.get_object_detection_results(
            batches, stream_ids, user_params
        )  # Preprocess, Object Detection

        alarms = {}
        now_target_cameras_for_tailgate = []

        if self.logging_flag:
            self.alarm_event_manager.bbox_location = raw_bbox_info_for_tailgate
            self.alarm_event_manager.letter_im = letter_im
            self.alarm_event_manager.roi = divided_roi_info

        for total_idx, origin_batch_idx in enumerate(category_info):  # Analysis Data
            # user_param = user_params[origin_batch_idx]
            target_camera_id = stream_ids[origin_batch_idx]
            origin_bboxs = raw_bbox_info_for_tailgate[target_camera_id]
            now_target_cameras_for_tailgate.append(target_camera_id)

            for roi_idx, roi in enumerate(
                divided_roi_info[total_idx]
            ):  # 한 카메라에 divided roi는 2개 이상. idx는
                y_scaled_bboxs = []
                for x1, y1, x2, y2 in origin_bboxs:
                    rho_location_info = xyxy2rhombus_for_topview(x1, y1, x2, y2)
                    in_roi_state = calc_intersect_for_topview(
                        [x1, y1, x2, y2], roi, rho_location_info
                    )
                    if in_roi_state:
                        y_scaled_bboxs.append(
                            [
                                rho_location_info[0][0],
                                rho_location_info[0][1],
                                rho_location_info[2][0],
                                rho_location_info[2][1],
                            ]
                        )
                position_list = self.alarm_event_manager._get_people_position_list(
                    vertical_info=(roi[:, 1].min(), roi[:, 1].max()),
                    bbox_infos=y_scaled_bboxs,
                )

                self.alarm_event_manager.update(
                    camera_id=target_camera_id, roi_idx=roi_idx, position_list=position_list
                )

        self.alarm_event_manager.get_alarm(
            now_target_cameras_for_tailgate, self.logging_flag
        )  # Tailgate Alarm Event Manager, Get Alarm

        for total_idx, origin_batch_idx in enumerate(category_info):  # Send Alarm
            state_dict = self.alarm_event_manager.event_state[stream_ids[origin_batch_idx]]
            for key, state in state_dict.items():
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
            i[USER_PARAM_KEY][CV_EVENT_KEY]['tailgate_cv'][OD_THRESHOLD_KEY] for i in user_params
        ]
        iou_thres = [
            i[USER_PARAM_KEY][CV_EVENT_KEY]['tailgate_cv'][IOU_THRESHOLD_KEY] for i in user_params
        ]

        nms_od_result = non_max_suppression(
            raw_od_result,
            conf_thres=conf_thres,
            agnostic=True,
            iou_thres=iou_thres,
            classes=OD_TARGET_CLASSES,
            max_det=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )  # 각 batch별 최대 인원을 7명으로 제한
        raw_bbox_info_for_tailgate = {}
        matched_info = {}

        for cropped_batch_idx, batch_idx_info in enumerate(categories_batches):
            now_bboxs = []
            for pred in nms_od_result[cropped_batch_idx]:
                x1, y1, x2, y2, conf, cls = pred
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                now_bboxs.append(pred[:4].detach().cpu().numpy())
            raw_bbox_info_for_tailgate[stream_ids[batch_idx_info]] = now_bboxs

        return (
            None,
            matched_info,
            raw_bbox_info_for_tailgate,
            categories_batches,
            divided_roi_info,
            letter_im,
        )
