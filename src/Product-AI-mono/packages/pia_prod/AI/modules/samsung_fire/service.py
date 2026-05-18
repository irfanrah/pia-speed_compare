import time
import csv
import os
import cv2
import numpy as np
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.utils.utils import get_roi_info

from pia_prod.AI.bases.service_base import ServiceBase
from pia.ai.device import load_model_backend

from pia_prod.AI.modules.samsung_fire.event import FireEventManager
from pia_prod.AI.global_config import USER_PARAM_KEY, CV_EVENT_KEY
from pia_prod.AI.modules.samsung_fire.config import (
    FIRE_CLS_MODEL_TRT_PATH,
    DEVICE,
    TRACK_TTL_FRAMES_DEFAULT,
    LABEL_MAP,
    FIRE_QUEUE_SIZE,
    ALARM_TRUE_RATIO_THRESH,
    ROI_RESIZE_SIZE,
    MIN_MOTION_RATIO,
    TRACK_IOU_THRESH,
    FIRE_BOX_EXPAND_RATIO,
    CLS_INPUT_SIZE,
)
from pia.vision.preprocessing import (
    batch_convert_colors,
    numba_batch_or,
    batch_color_filter,
    resize_batches,
)
from pia.vision.postprocessing import (
    find_contours,
    filter_bboxes_by_motion,
    update_tracked_boxes,
    expand_batches_bboxes,
    crop_batches_bboxes,
)
from pia_prod.AI.modules.samsung_fire.debug_utils import save_fire_snapshot
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class FireService(ServiceBase):
    def __init__(self, analysis_data_queue, save_csv: bool = True):
        super().__init__(analysis_data_queue)
        self.save_csv = save_csv

    def _init_values(self):
        self.frame_cnt = 0
        self.tracked_boxes_dict = {}
        self.last_ts_dict = {}

        self._csv_file = None
        self._csv_writer = None

    def _load_model(self):
        self.fire_cls_model = PiaONNXTensorRTModel(
            FIRE_CLS_MODEL_TRT_PATH, device=load_model_backend(DEVICE)
        )

    def _load_event_manager(self):
        return FireEventManager()

    def _get_csv_writer(self, sid: str):
        if not getattr(self, "save_csv", False):
            return None
        if self._csv_writer is None:
            dirpath = os.path.join("logs", "csv")
            os.makedirs(dirpath, exist_ok=True)
            path = os.path.join(dirpath, f"{sid}.csv")
            f = open(path, "w", newline="")
            w = csv.writer(f)
            w.writerow(["frame", "output", "result", "alarm"])
            self._csv_file = f
            self._csv_writer = w
        return self._csv_writer

    def close_csvs(self) -> None:
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file = None
        self._csv_writer = None

    def _estimate_fps(self, sid: str) -> int:
        now = time.time()
        last = self.last_ts_dict.get(sid)
        self.last_ts_dict[sid] = now
        if last is None:
            return 0
        delta = now - last
        return int(round(1 / delta)) if delta > 0 else 0

    @staticmethod
    def _get_color_filter_values(user_params, color_map):
        result = []
        for user_param in user_params:
            for model in user_param[USER_PARAM_KEY][CV_EVENT_KEY].values():
                lower = np.array(model[f"lower_{color_map}"], dtype=np.uint8)
                upper = np.array(model[f"upper_{color_map}"], dtype=np.uint8)
                result.append([lower, upper])
        return result

    @staticmethod
    def _get_bbox_min_area(user_params):
        result = []
        for user_param in user_params:
            for model in user_param[USER_PARAM_KEY][CV_EVENT_KEY].values():
                result.append(model["bbox_min_area"])
        return result

    def _compute_metrics(self, sid: str, thresh_cnt: int):
        preds = self.alarm_event_manager.batch_preds_dict[sid]
        output = "fire" if any(lbl == "fire" for lbl in preds) else "normal"
        dq = self.alarm_event_manager.event_dict[sid]
        result = 1 if sum(dq) >= thresh_cnt else 0
        alarm_state = self.alarm_event_manager.event_status[sid]
        return output, result, alarm_state

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        # 0) 프레임 카운트
        self.frame_cnt += 1 if self.save_csv else 0
        sid = stream_ids[0]

        # 1) CSV writer & 기본값 준비
        writer = self._get_csv_writer(sid) if self.save_csv else None
        output = "normal"
        result = self.alarm_event_manager.event_status.get(sid, 0)
        alarm_state = self.alarm_event_manager.event_status.get(sid, 0)
        thresh_cnt = int(FIRE_QUEUE_SIZE * ALARM_TRUE_RATIO_THRESH)

        # 2) ROI 처리
        rois = get_roi_info(user_params)
        roi_erased, _ = self.roi_manager.erase_roi(batches, rois)
        roi_erased = [
            cv2.resize(roi, ROI_RESIZE_SIZE, interpolation=cv2.INTER_LINEAR) for roi in roi_erased
        ]

        # 3) 색/모션 마스크
        hsv_values = self._get_color_filter_values(user_params, "hsv")  # hsv 값 파싱
        ycc_values = self._get_color_filter_values(user_params, "ycc")  # YCrCb 값 파싱
        converted_images = batch_convert_colors(
            roi_erased, conditions=["rgb", "hsv", "ycrcb", "gray"]
        )
        batch_rgb = converted_images["rgb"]
        batch_hsv = converted_images["hsv"]
        batch_ycc = converted_images["ycrcb"]
        batch_gray = converted_images["gray"]

        hsv_masks = batch_color_filter(batch_hsv, hsv_values)
        ycc_masks = batch_color_filter(batch_ycc, ycc_values)
        color_masks = numba_batch_or(hsv_masks, ycc_masks)
        motion_masks = self.alarm_event_manager.make_motion_mask(batch_gray, stream_ids)

        # 4) 컨투어 → bbox(색상+모션)
        bbox_min_areas = self._get_bbox_min_area(user_params)  # bbox 최소 크기 값 파싱
        color_contours = find_contours(color_masks, bbox_min_areas=bbox_min_areas)
        detected_bboxes = [
            filter_bboxes_by_motion(
                color_contours[i], motion_masks[i], min_motion_ratio=MIN_MOTION_RATIO
            )
            for i in range(len(stream_ids))
        ]

        # 5) IoU+TTL 트래킹
        for i, sid_i in enumerate(stream_ids):
            fps = self._estimate_fps(sid_i)
            ttl = fps * 1 if fps else TRACK_TTL_FRAMES_DEFAULT
            prev = self.tracked_boxes_dict.get(sid_i, [])
            self.tracked_boxes_dict[sid_i] = update_tracked_boxes(
                detected_bboxes[i], prev, ttl_frames=ttl, iou_thresh=TRACK_IOU_THRESH
            )

        # 6) bbox 확장 & crop
        expanded_bboxes = expand_batches_bboxes(
            detected_bboxes,
            max_w=ROI_RESIZE_SIZE[0],
            max_h=ROI_RESIZE_SIZE[1],
            top_ratio=FIRE_BOX_EXPAND_RATIO,
            bottom_ratio=FIRE_BOX_EXPAND_RATIO,
            left_ratio=FIRE_BOX_EXPAND_RATIO,
            right_ratio=FIRE_BOX_EXPAND_RATIO,
        )

        crop_batches = crop_batches_bboxes(expanded_bboxes, batch_rgb)

        # 7) 모델 추론 (bbox 있을 때만)
        if any(len(bbox) for bbox in crop_batches):
            flat_crops = self.alarm_event_manager.make_inference_batches(crop_batches, stream_ids)
            resized_crop_bboxes, _ = resize_batches(flat_crops, device=DEVICE, shape=CLS_INPUT_SIZE)

            preds_tensor = self.fire_cls_model(resized_crop_bboxes)
            idx = preds_tensor.argmax(dim=1)
            raw_labels = [f'class{i}' for i in idx.cpu().tolist()]
            pred_labels = [LABEL_MAP.get(lbl, lbl) for lbl in raw_labels]

            alarms = self.alarm_event_manager.update(pred_labels, stream_ids)

            output, result, alarm_state = self._compute_metrics(sid, thresh_cnt)
        else:
            alarms = self.alarm_event_manager([], stream_ids)

        # 8) CSV에 기록
        if writer:
            writer.writerow([self.frame_cnt, output, result, alarm_state])
            self._csv_file.flush()

        if self.logging_flag:
            for sid_i, is_start in alarms.items():
                if is_start:
                    batch_idx = stream_ids.index(sid_i)
                    output, result, _ = self._compute_metrics(sid_i, thresh_cnt)
                    save_fire_snapshot(
                        image=roi_erased[batch_idx].copy(),
                        stream_id=sid_i,
                        output=output,
                        result=result,
                        bboxes=expanded_bboxes[batch_idx],
                        classes=self.alarm_event_manager.batch_preds_dict[sid_i],
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
