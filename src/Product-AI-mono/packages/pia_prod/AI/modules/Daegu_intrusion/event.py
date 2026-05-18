from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.Daegu_intrusion.config import (
    INTRUSION_ALARM_DURATION,
    INTRUSION_QUEUE_SIZE,
)
from pia.vision.postprocessing.bbox import calc_intersect


class IntrusionEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "intrusion_cv"
        self.alarm_duration = INTRUSION_ALARM_DURATION
        self.duration_queue = defaultdict(lambda: deque(maxlen=(INTRUSION_QUEUE_SIZE)))
        self.event_status = defaultdict(int)

    def update(self, results, stream_ids, rois):
        # ROI 내에 들어갔는지 확인
        alarms = []
        for tracks, stream_id in zip(results, stream_ids):
            cnt = 0
            for track in tracks:
                if len(track) != 0:
                    is_in_roi = calc_intersect(
                        track[:4], rois[stream_id]["after_letterbox_calc_origin_roi"]
                    )
                    if is_in_roi:
                        cnt = 1
                        break
            self.duration_queue[stream_id].append(cnt)
        alarms = self.check_alarm_duration()
        return alarms

    def check_alarm_duration(self):
        alarms = {}
        for key, value in self.duration_queue.items():
            before_status = self.event_status[key]
            is_over_queue = int(sum(value) >= INTRUSION_ALARM_DURATION)
            now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
            self.event_status[key] = now_status
            if now_status in [1, 3]:
                alarms[key] = [self.EVENT_STATUS_DICT[now_status], None]
        return alarms
