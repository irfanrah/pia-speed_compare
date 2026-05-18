from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.vehicle_reverse.config import (
    EVENT_QUEUE_SIZE,
    EVENT_QUEUE_THRESHOLD,
    CATEGORY_NAME,
)


class VehicleReverseEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = CATEGORY_NAME
        self.alert_message = "ALERT: Vehicle reverse detected"
        self.duration_queue = defaultdict(lambda: deque(maxlen=EVENT_QUEUE_SIZE))
        self.event_status = defaultdict(int)

    def update(self, events):
        alarms = []
        for event in events:
            cnt = 0
            if event['is_wrong']:
                cnt = 1
            self.duration_queue[event["stream_id"]].append(cnt)
        alarms = self.check_alarm_duration()
        return alarms

    def check_alarm_duration(self):
        alarms = {}
        for key, value in self.duration_queue.items():
            before_status = self.event_status[key]
            is_over_queue = int(sum(value) >= EVENT_QUEUE_THRESHOLD)
            now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
            self.event_status[key] = now_status
            if now_status in [1, 3]:
                alarms[key] = [self.EVENT_STATUS_DICT[now_status], None]
        return alarms
