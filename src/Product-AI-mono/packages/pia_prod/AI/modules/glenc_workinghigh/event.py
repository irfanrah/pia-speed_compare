from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.glenc_workinghigh.config import (
    WORKINGHIGH_ALARM_DURATION,
    WORKINGHIGH_QUEUE_SIZE,
)
from pia_prod.AI.modules.glenc_workinghigh.func import any_bbox_corner_in_roi


class WorkinghighEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "workinghigh_cv"
        self.alarm_duration = WORKINGHIGH_ALARM_DURATION
        self.duration_queue = defaultdict(lambda: deque(maxlen=(WORKINGHIGH_QUEUE_SIZE)))
        self.event_status = defaultdict(int)

    def update(self, results, stream_ids, rois):
        # bbox 의 4 corner 중 하나라도 RoI 안에 있는지 확인.
        # (Daegu_intrusion 의 calc_intersect 는 4 corner 가 모두 RoI 안일 때만
        # True 를 반환했지만, 고소작업은 일부만 들어와도 이벤트로 잡아야 함.)
        for tracks, stream_id in zip(results, stream_ids):
            cnt = 0
            for track in tracks:
                if len(track) != 0:
                    is_in_roi = any_bbox_corner_in_roi(
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
            is_over_queue = int(sum(value) >= WORKINGHIGH_ALARM_DURATION)
            now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
            self.event_status[key] = now_status
            if now_status in [1, 3]:
                alarms[key] = [self.EVENT_STATUS_DICT[now_status], None]
        return alarms
