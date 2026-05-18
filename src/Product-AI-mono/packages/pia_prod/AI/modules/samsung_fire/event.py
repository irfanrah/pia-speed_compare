import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.samsung_fire.config import (
    FIRE_QUEUE_SIZE,
    ALARM_TRUE_RATIO_THRESH,
    LOWER_HSV,
    UPPER_HSV,
    LOWER_YCrCb,
    UPPER_YCrCb,
    COLOR_FILTER_KERNEL_SIZE,
    MOTION_FILTER_KERNEL_SIZE,
    MOTION_THRESH_MIN,
    MOTION_THRESH_MAX,
)


class FireEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "fire_cv"
        self.upper_hsv_array = np.array(UPPER_HSV)
        self.lower_hsv_array = np.array(LOWER_HSV)
        self.upper_YCrCb_array = np.array(UPPER_YCrCb)
        self.lower_YCrCb_array = np.array(LOWER_YCrCb)

        self.event_dict = defaultdict(lambda: deque(maxlen=FIRE_QUEUE_SIZE))
        self.event_status = defaultdict(int)
        self.batch_index_dict = defaultdict(int)
        self.prev_gray = defaultdict(lambda: None)
        self.batch_preds_dict = defaultdict(list)

        self.motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MOTION_FILTER_KERNEL_SIZE)
        self.color_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, COLOR_FILTER_KERNEL_SIZE)

    def make_inference_batches(self, crops: List[List[np.ndarray]], sids: List[str]):
        flat = []
        for crops_per_sid, sid in zip(crops, sids):
            self.batch_index_dict[sid] = len(crops_per_sid)
            flat.extend(crops_per_sid)
        return flat

    def make_motion_mask(self, gray_batches: List[np.ndarray], sids: List[str]):
        masks = []
        for gray, sid in zip(gray_batches, sids):
            prev = self.prev_gray[sid]
            if prev is None:
                masks.append(np.zeros_like(gray))
            else:
                diff = cv2.absdiff(gray, prev)
                _, m1 = cv2.threshold(diff, MOTION_THRESH_MIN, 255, cv2.THRESH_BINARY)
                _, m2 = cv2.threshold(diff, MOTION_THRESH_MAX, 255, cv2.THRESH_BINARY_INV)
                m = cv2.bitwise_and(m1, m2)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.motion_kernel, iterations=1)
                m = cv2.dilate(m, self.motion_kernel, iterations=1)
                masks.append(m)
            self.prev_gray[sid] = gray
        return masks

    def update(self, batches_pred: List[str], stream_ids: List[str]):
        cnt = 0
        for sid in stream_ids:
            value = self.batch_index_dict[sid]
            preds = batches_pred[cnt : cnt + value]
            cnt += value

            self.batch_preds_dict[sid] = preds
            self.event_dict[sid].append(int(any(lbl == "fire" for lbl in preds)))
        return self.check_alarm_duration()

    def check_alarm_duration(self):
        alarms = {}
        thresh_cnt = int(FIRE_QUEUE_SIZE * ALARM_TRUE_RATIO_THRESH)

        for sid, dq in self.event_dict.items():
            now_status = sum(dq) >= thresh_cnt
            before_status = self.event_status[sid]

            self.event_status[sid] = self.STATUS_TRANSITION[before_status][now_status]
            if self.event_status[sid] in [1, 3]:  # start or alarm
                alarms[sid] = [self.EVENT_STATUS_DICT[self.event_status[sid]], None]

        return alarms
