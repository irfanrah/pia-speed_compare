from collections import deque, defaultdict
from typing import List

from pia.utils.devtools.debug_tools import print_only_debug_mode
from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.soil_qwen35.config import (
    QUEUE_SIZE,
    ALARM_DURATION_THRESHOLD,
)


class SoilQwen35EventManager(EventBase):
    def __init__(self):
        super().__init__(alarm_duration=ALARM_DURATION_THRESHOLD)
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=QUEUE_SIZE)))
        self.event_status = defaultdict(lambda: defaultdict(int))

    def update(self, responses: List[str], categories_per_stream: List[List[str]], stream_ids: List[str]):
        resp_idx = 0
        for categories, stream_id in zip(categories_per_stream, stream_ids):
            for cat in categories:
                response = responses[resp_idx]
                resp_idx += 1
                is_positive = self.parse_yes_no(response)
                print_only_debug_mode(
                    f"stream_id: {stream_id}\tcategory: {cat}\tresponse: {response}\tpositive: {is_positive}"
                )
                self.duration_queue[stream_id][cat].append(1 if is_positive else 0)
        return self.check_alarm_duration()

    def check_alarm_duration(self):
        alarms = {}
        for stream_id, cat_dict in self.duration_queue.items():
            for category_id, value in cat_dict.items():
                before_status = self.event_status[stream_id][category_id]
                is_over_queue = int(sum(value) >= ALARM_DURATION_THRESHOLD)
                now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
                self.event_status[stream_id][category_id] = now_status
                if now_status in [1, 3]:
                    alarms[stream_id] = [self.EVENT_STATUS_DICT[now_status], category_id]
        return alarms

    @staticmethod
    def parse_yes_no(response) -> bool:
        if not isinstance(response, str):
            return False
        return response.strip().lower() in ("yes", "1")
