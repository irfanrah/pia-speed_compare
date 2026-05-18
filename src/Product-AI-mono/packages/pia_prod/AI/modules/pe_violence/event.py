from typing import Dict, List
from pia_prod.AI.bases.event_base import EventBase
from collections import deque, defaultdict


class PVEventManager(EventBase):
    """
    Alarm queue mechanism for violence detection.
    - Maintains a fixed-size queue of recent predictions per stream.
    - An alarm triggers when >= alarm_threshold predictions in the queue are anomalous.
    - Setting both alarm_queue_size and alarm_threshold to 1 disables smoothing.
    """

    def __init__(self, alarm_queue_size: int, alarm_threshold: int):
        super().__init__(alarm_duration=alarm_threshold)
        self.alarm_queue_size = alarm_queue_size
        self.alarm_threshold = alarm_threshold
        self.duration_queue = defaultdict(lambda: deque(maxlen=self.alarm_queue_size))
        self.event_status = defaultdict(int)

    def update(
        self,
        predicts: Dict[str, bool],
        stream_ids: List[str],
    ):
        for stream_id, predict in zip(stream_ids, predicts):
            is_violence = bool(predict)
            self.duration_queue[stream_id].append(is_violence)
        return self.check_alarm_duration(stream_ids)

    def check_alarm_duration(self, stream_ids):
        alarms = {}
        for stream_id in stream_ids:
            before_status = self.event_status[stream_id]
            now_status = sum(self.duration_queue[stream_id]) >= self.alarm_threshold
            final_status = self.STATUS_TRANSITION[before_status][now_status]
            self.event_status[stream_id] = final_status
            if final_status in [1, 3]:
                alarms[stream_id] = [self.EVENT_STATUS_DICT[final_status], None]
        return alarms
