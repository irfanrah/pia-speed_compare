from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.utils.utils import get_roi_info

# from pia_prod.AI.modules.yeonsei_smoke.debug_utils import save_smoke_snapshot
from pia_prod.AI.modules.yeonsei_smoke.event import SmokeEventManager
from pia.vision.preprocessing import (
    batch_convert_colors,
    batch_color_filter,
    numba_batch_and,
    resize_batches,
    LetterBox,
)
from pia.vision.postprocessing import find_contours, expand_batches_bboxes, crop_batches_bboxes

from pia.ai.device import load_model_backend
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.global_config import USER_PARAM_KEY, CV_EVENT_KEY
from pia_prod.AI.modules.yeonsei_smoke.config import (
    SMOKE_CLS_MODEL_TRT_PATH,
    DEVICE,
    COLOR_FILTER_OPS_CONDITION,
    SMOKE_BOX_EXPAND_RATIO,
    CLS_INPUT_SIZE,
)
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)
import numpy as np


# 연기, 쓰러짐 검출을 위한 service
class SmokeService(ServiceBase):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)

    def _init_values(self):
        self.letter_box = LetterBox(CLS_INPUT_SIZE)

    def _load_model(self):
        # smoke cls model
        self.smoke_cls_model = PiaONNXTensorRTModel(
            SMOKE_CLS_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        return SmokeEventManager()

    @staticmethod
    def _get_color_filter_values(user_params):
        result = []
        for user_param in user_params:
            for model in user_param[USER_PARAM_KEY][CV_EVENT_KEY].values():
                lower_hsv = np.array(model["lower_hsv"], dtype=np.uint8)
                upper_hsv = np.array(model["upper_hsv"], dtype=np.uint8)
                result.append([lower_hsv, upper_hsv])
        return result

    @staticmethod
    def _get_bbox_min_area(user_params):
        result = []
        for user_param in user_params:
            for model in user_param[USER_PARAM_KEY][CV_EVENT_KEY].values():
                result.append(model["bbox_min_area"])
        return result

    @staticmethod
    def _get_cls_threshold(user_params):
        result = []
        for user_param in user_params:
            for model in user_param[USER_PARAM_KEY][CV_EVENT_KEY].values():
                result.append(model["cls_threshold"])
        return result

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        rois = get_roi_info(user_params)  # ROI 정보 파싱
        hsv_values = self._get_color_filter_values(user_params)  # hsv 값 파싱
        bbox_min_areas = self._get_bbox_min_area(user_params)  # bbox 최소 크기 값 파싱
        cls_thresholds = self._get_cls_threshold(user_params)
        # resize
        roi_erased_batches, _ = self.roi_manager.erase_roi(batches, rois)

        # cvt color
        converted_batches = batch_convert_colors(
            roi_erased_batches, conditions=["rgb", "hsv", "gray"]
        )
        rgb_batches = converted_batches["rgb"]
        hsv_batches = converted_batches["hsv"]
        gray_batches = converted_batches["gray"]

        # color mask
        color_filtering_masks = batch_color_filter(
            hsv_batches, hsv_values, operations=[COLOR_FILTER_OPS_CONDITION]
        )

        # motion mask
        motion_filtering_masks = self.alarm_event_manager.make_motion_mask(gray_batches, stream_ids)

        # bitwise and
        concat_masks = numba_batch_and(color_filtering_masks, motion_filtering_masks)

        # filtered contours
        contours = find_contours(concat_masks, bbox_min_areas=bbox_min_areas)

        # make expanded bboxes
        expanded_bboxes = expand_batches_bboxes(
            contours,
            max_w=CLS_INPUT_SIZE[0],
            max_h=CLS_INPUT_SIZE[1],
            top_ratio=SMOKE_BOX_EXPAND_RATIO,
            bottom_ratio=SMOKE_BOX_EXPAND_RATIO,
            left_ratio=SMOKE_BOX_EXPAND_RATIO,
            right_ratio=SMOKE_BOX_EXPAND_RATIO,
        )

        # crop bboxes
        crop_bboxes = crop_batches_bboxes(
            expanded_bboxes,
            rgb_batches,
        )

        if any(len(inner) != 0 for inner in crop_bboxes):
            re_batches = self.alarm_event_manager.make_inference_batches(crop_bboxes, stream_ids)
            preprocessed_crop_bboxes, _ = resize_batches(
                re_batches, device=DEVICE, letterbox_instance=self.letter_box
            )
            # classification
            batches_pred = self.smoke_cls_model(preprocessed_crop_bboxes)

            # make alarm
            self.alarm_event_manager(batches_pred, stream_ids, cls_thresholds)
            self.alarm_event_manager.check_alarm_duration(self.alarm_event_manager.event_dict)

            alarms = self.alarm_event_manager.get_alarm_dict()

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

        else:
            self.alarm_event_manager([], stream_ids, cls_thresholds)
            return None
