from pia_prod.AI.bases.event_base import EventBase
from typing import List
from collections import defaultdict, deque
import numpy as np
import cv2
from pia_prod.AI.modules.yeonsei_smoke.config import (
    LOWER_HSV,
    UPPER_HSV,
    COLOR_FILTER_KERNEL_SIZE,
    BG_KERNEL1_SIZE,
    BG_KERNEL2_SIZE,
    NUM_HISTORY,
    VAR_THRESHOLD,
    LEARNING_RATE,
    SMOKE_QUEUE_SIZE,
    SMOKE_ALARM_DURATION,
    FRAME_SKIP_COUNT,
    RESIZE_SIZE,
)


class SmokeEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "smoke_cv"
        self.upper_hsv_array = np.array(UPPER_HSV)
        self.lower_hsv_array = np.array(LOWER_HSV)
        self.color_filter_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, COLOR_FILTER_KERNEL_SIZE
        )
        self.BG_filter = defaultdict(lambda: BackgroundSubtractor())
        self.BG_cnt_dict = defaultdict(int)
        self.frame_cnt = 0
        self.event_status = defaultdict(int)
        self.batch_index_dict = defaultdict(int)
        self.event_dict = defaultdict(lambda: deque(maxlen=SMOKE_QUEUE_SIZE))
        self.event_states = defaultdict(int)
        self.zero_mask = np.zeros(RESIZE_SIZE, dtype=np.uint8)
        self.batch_preds_dict = defaultdict()

    def update(self, batches_pred, stream_ids, cls_thresholds):
        cnt = 0
        for key, threshold in zip(stream_ids, cls_thresholds):
            value = self.batch_index_dict[key]
            is_smoke = 0
            if len(batches_pred):
                self.batch_preds_dict[key] = batches_pred[cnt : cnt + value][:, 1]
                if any(batches_pred[cnt : cnt + value][:, 1] > threshold):
                    is_smoke = 1
            self.event_dict[key].append(is_smoke)
            cnt += value

    def check_alarm_duration(self, event_dict: defaultdict):
        for key, value in event_dict.items():
            before_state = self.event_status[key]
            if sum(value) >= SMOKE_ALARM_DURATION:
                after_state = 1  # Event True
            else:
                after_state = 0  # Event False

            now_state = self.STATUS_TRANSITION[before_state][after_state]
            self.event_status[key] = now_state

    def make_inference_batches(self, batches, stream_ids):
        result = []
        for batch, stream_id in zip(batches, stream_ids):
            self.batch_index_dict[stream_id] = len(batch)
            if len(batch):
                result += batch
        return result

    def __imwrite_batches(self, batches, prefix=""):
        for i, im in enumerate(batches):
            cv2.imwrite(f"logs/frame_{prefix}_{self.frame_cnt}_{i}.jpg", im)

    def make_motion_mask(self, batches: List[np.ndarray], stream_ids) -> List[np.ndarray]:
        result = []
        for gray_frame, stream_id in zip(batches, stream_ids):
            mask = self.BG_filter[stream_id].apply(gray_frame)
            if self.BG_cnt_dict[stream_id] > FRAME_SKIP_COUNT:
                result.append(mask)
            else:
                self.BG_cnt_dict[stream_id] += 1
                result.append(self.zero_mask)
        return result

    def get_alarm_dict(self, logging_state: bool = False) -> dict:
        """
        Returns the current event status dictionary.
        """
        alarms = dict()
        for stream_id, state in self.event_status.items():
            if logging_state and state in [1, 2]:
                # save_smoke_snapshot(roi_erased_batches[batch_idx],
                #     stream_id,
                #     category_name,
                #     self.smoke_event_manager.batch_preds_dict[stream_id],
                #     expanded_bboxes[batch_idx])
                pass  # 추후 스냅샷 저장 필요할 시 해당 코드 작성 필요
            if state not in [1, 3]:
                continue  # 막시작한 이벤트(1)나, 막 종료한 이벤트(3)가 아니면 넘김
            is_start = True if state == 1 else False
            alarms[stream_id] = [is_start, None]

        return alarms


class BackgroundSubtractor:
    def __init__(self):
        self.history = NUM_HISTORY
        self.varThreshold = VAR_THRESHOLD
        self.learningRate = LEARNING_RATE
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BG_KERNEL1_SIZE)
        self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BG_KERNEL2_SIZE)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=False,
        )

    def apply(self, frame):
        motion_mask = self.fgbg.apply(frame)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, self.kernel2, iterations=1)
        return motion_mask
