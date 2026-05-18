from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.ax_fall.config import (
    FALL_QUEUE_THRESHOLD,
    FALL_QUEUE_SIZE,
)


class FallEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "fallhazard_cv"
        self.alarm_duration = FALL_QUEUE_THRESHOLD
        self.duration_queue = defaultdict(lambda: deque(maxlen=(FALL_QUEUE_SIZE)))
        self.event_status = defaultdict(int)

    def update(self, results, stream_ids, rois):
        # ROI 내에 들어갔는지 확인
        alarms = []
        for tracks, stream_id in zip(results, stream_ids):
            cnt = 1 if len(tracks) else 0
            self.duration_queue[stream_id].append(cnt)
        alarms = self.check_alarm_duration(stream_ids)
        return alarms

    def check_alarm_duration(self, target_cameras: list):
        alarms = {}
        for target_camera in target_cameras:
            value = self.duration_queue[target_camera]
            before_status = self.event_status[target_camera]
            is_over_queue = int(sum(value) >= FALL_QUEUE_THRESHOLD)
            now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
            self.event_status[target_camera] = now_status
            if now_status in [1, 3]:
                alarms[target_camera] = [self.EVENT_STATUS_DICT[now_status], None]
        return alarms
