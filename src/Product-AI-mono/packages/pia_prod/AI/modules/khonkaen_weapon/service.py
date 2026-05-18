from typing import List, Dict
import numpy as np
from pia_prod.AI.bases.service_base import ServiceBase
from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia_prod.AI.modules.khonkaen_weapon.config import (
    DEVICE,
    PERSON_DETECTION_MODEL_TRT_PATH,
    WEAPON_DETECTION_MODEL_TRT_PATH,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_NMS_THRESHOLD,
    OD_INPUT_SIZE,
    OD_TARGET_CLASSES,
    WEAPON_DETECTION_CONFIDENCE_THRESHOLD,
    TARGET_CATEGORY_INDEX,
    TOP_MARGIN_RATIO,
    LEFT_MARGIN_RATIO,
    RIGHT_MARGIN_RATIO,
    BOTTOM_MARGIN_RATIO,
    LIMITED_NUM_OF_CAMERA,
    OD_CONFIDENCE_THRESHOLD,
)

from pia_prod.AI.modules.khonkaen_weapon.event import WeaponEventManager
from pia_prod.AI.modules.khonkaen_weapon.roi_manager import WeaponRoIManager
from pia_prod.AI.modules.khonkaen_weapon.debug_utils import save_snapshot_for_weapon

from pia.vision.preprocessing.resize import LetterBoxTorch
from pia.vision.postprocessing.bbox import expand_bbox
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


class WeaponService(ServiceBase):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.alert_message = "Weapon Detected!!!"
        self.save_video = False  # Debugging 용도, test_weapon_1batch에서 사용
        self.video_manager = None  # Debugging 용도, test_weapon_1batch에서 사용
        self.detected_counter = 0  # Debugging 용도, test_weapon_1batch에서 사용

    def __del__(self):
        self.del_model()

    def _init_values(self):
        self.pgie_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA,
            target_size=OD_INPUT_SIZE,
            device=DEVICE,
        )
        self.sgie_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_PERSON_PER_CAMERA * LIMITED_NUM_OF_CAMERA,
            target_size=OD_INPUT_SIZE,
            device=DEVICE,
        )

    def _load_roi_manager(self):
        return WeaponRoIManager()

    def _load_model(self):
        self.pgie_detection_instance = PiaONNXTensorRTModel(
            model_path=PERSON_DETECTION_MODEL_TRT_PATH,
            device=DEVICE,
        )
        self.sgie_detection_instance = PiaONNXTensorRTModel(
            model_path=WEAPON_DETECTION_MODEL_TRT_PATH,
            device=DEVICE,
        )

    def _load_event_manager(self):
        return WeaponEventManager()

    def del_model(self):
        """모델을 삭제합니다."""
        if hasattr(self, "pgie_detection_instance"):
            free_autobackend(self.pgie_detection_instance)

        if hasattr(self, "sgie_detection_instance"):
            free_autobackend(self.sgie_detection_instance)

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        (
            cropped_images,
            matched_info,
            self.total_bbox_info,  # for draw snapshot
            cropped_batches,  # for debug
        ) = self.get_pgie_results(batches=batches, stream_ids=stream_ids, user_params=user_params)

        sgie_results = self.get_sgie_results(
            cropped_images=cropped_images,
            matched_info=matched_info,
            stream_ids=stream_ids,
            user_params=user_params,
        )

        self.alarm_event_manager.update(sgie_results, stream_ids)
        alarms = self.alarm_event_manager.get_alarms(stream_ids)

        if self.logging_flag:
            save_snapshot_for_weapon(
                total_bbox_info=self.total_bbox_info,
                cropped_batches=cropped_batches,
                stream_ids=stream_ids,
                alarms=alarms,
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
        user_params: List[Dict],
    ):
        # 추후 nms에서 batch별로 conf, iou threshold를 다르게 주기 위해 리스트로 변경
        # conf_thres = []
        # iou_thres = []
        # for i in user_params:
        #     cv_event = i[USER_PARAM_KEY][CV_EVENT_KEY]
        #     for key in WEAPON_CV_CATEGORY:
        #         if key in cv_event:
        #             conf_thres.append(cv_event[key][OD_THRESHOLD_KEY])
        #             iou_thres.append(cv_event[key][IOU_THRESHOLD_KEY])
        #             break

        # Preprocess 1 - Crop with RoI
        cropped_batches = self.roi_manager.process_batches_with_roi(
            batches=batches, user_params=user_params
        )

        # Preprocess 2 - Letterbox & Normalize
        resized_images = self.pgie_letterbox_instance(cropped_batches)
        preprocess_images = resized_images / 255.0

        # Inference model
        raw_od_results = self.pgie_detection_instance(preprocess_images)

        # Postprocess
        # TODO: conf_thres, iou_thres 리스트로 유저에 의해 변경될 수 있도록 변경
        nms_od_results = torch_non_max_suppression(
            prediction=raw_od_results,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=OD_TARGET_CLASSES,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )

        total_bbox_info = []
        cropped_images = []
        matched_info = {}
        for cropped_batch_index, batch_index_info in enumerate(range(len(batches))):
            cnt = 0
            now_bbox_info = []
            max_w, max_h = (
                cropped_batches[cropped_batch_index].shape[2],
                cropped_batches[cropped_batch_index].shape[1],
            )
            for pred in nms_od_results[cropped_batch_index]:
                x1, y1, x2, y2, conf, cls = pred
                x1, y1, x2, y2 = expand_bbox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    max_w=max_w,
                    max_h=max_h,
                    top_ratio=TOP_MARGIN_RATIO,
                    left_ratio=LEFT_MARGIN_RATIO,
                    right_ratio=RIGHT_MARGIN_RATIO,
                    bottom_ratio=BOTTOM_MARGIN_RATIO,
                )
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue  # crop 불가능한 bbox 제외

                origin_bbox = modified2origin_coordinate(
                    xyxy=(x1, y1, x2, y2),
                    now_shape=OD_INPUT_SIZE,
                    original_shape=cropped_batches[batch_index_info].shape[1:3],
                )
                origin_bbox.extend([conf.item(), int(cls.item())])
                now_bbox_info.append(origin_bbox)
                x1, y1, x2, y2 = origin_bbox[:4]
                cropped_image = cropped_batches[batch_index_info][:, y1:y2, x1:x2]
                cropped_images.append(cropped_image)
                cnt += 1
            matched_info[stream_ids[batch_index_info]] = cnt
            total_bbox_info.append(now_bbox_info)
        return (cropped_images, matched_info, total_bbox_info, cropped_batches)

    def get_sgie_results(
        self,
        cropped_images: List[np.array],
        matched_info: Dict,
        stream_ids: List[str],
        user_params: List[Dict],
    ) -> List[set]:
        result = []
        if len(cropped_images) == 0:
            return [{0}] * len(matched_info)

        resized_images = self.sgie_letterbox_instance(cropped_images)
        resized_images = resized_images / 255.0
        raw_sgie_results = self.sgie_detection_instance(resized_images)

        nms_od_results = torch_non_max_suppression(
            prediction=raw_sgie_results,
            conf_thres=WEAPON_DETECTION_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=TARGET_CATEGORY_INDEX,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )

        if self.logging_flag:
            cnt = 0
            for stream_index in range(len(user_params)):  # 0,1,2...
                now_key = stream_ids[stream_index]
                now_num_batches = matched_info[now_key]
                target_od_results = nms_od_results[cnt : cnt + now_num_batches]
                for target_idx, target in enumerate(target_od_results):
                    if len(target) != 0:
                        for pred in target:
                            x1, y1, x2, y2 = map(int, pred[:4].tolist())
                            conf = pred[4].item()
                            cls = int(pred[5].item())
                            related_coordinates = modified2origin_coordinate(
                                xyxy=(x1, y1, x2, y2),
                                now_shape=OD_INPUT_SIZE,
                                original_shape=cropped_images[cnt + target_idx].shape[1:3],
                            )
                            real_x1 = (
                                self.total_bbox_info[stream_index][target_idx][0]
                                + related_coordinates[0]
                            )
                            real_y1 = (
                                self.total_bbox_info[stream_index][target_idx][1]
                                + related_coordinates[1]
                            )
                            real_x2 = (
                                self.total_bbox_info[stream_index][target_idx][0]
                                + related_coordinates[2]
                            )
                            real_y2 = (
                                self.total_bbox_info[stream_index][target_idx][1]
                                + related_coordinates[3]
                            )

                            self.total_bbox_info[stream_index].append(
                                [real_x1, real_y1, real_x2, real_y2, conf, cls]
                            )
                cnt += now_num_batches

        cnt = 0
        for stream_index in range(len(user_params)):  # 0,1,2...
            now_key = stream_ids[stream_index]
            now_num_batches = matched_info[now_key]
            if now_num_batches == 0:
                result.append({0})
                continue

            detected_class_idx = set()
            target_od_results = nms_od_results[cnt : cnt + now_num_batches]

            for t in target_od_results:  # 카테고리 정보만 set 자료형으로 추출하여 배치마다 저장
                cats = t[:, 5].tolist()
                detected_class_idx.update(int(c) for c in cats)

            result.append(detected_class_idx)
            cnt += now_num_batches

        return result
