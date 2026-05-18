from pia_prod.AI.bases.event_base import EventBase
from collections import deque, defaultdict
from typing import List
import json
import re
from pia.utils.devtools.debug_tools import print_only_debug_mode
from pia_prod.AI.modules.gangnam_falldown.config import (
    QUEUE_SIZE,
    ALARM_DURATION_THRESHOLD,
    MODEL_OUTPUTS,
    FALLDOWN_CATEGORY,
    FIRE_CATEGORY,
)


class VQAFalldownEventManager(EventBase):
    def __init__(self):
        super().__init__(
            # queue_size=QUEUE_SIZE,
            alarm_duration=ALARM_DURATION_THRESHOLD
        )
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=(QUEUE_SIZE))))
        self.event_status = defaultdict(lambda: defaultdict(int))

    def update(self, responses: List[str], categories_per_stream: List[str], stream_ids: List[str]):
        # categories_per_stream : [[cat1, cat2, cat3], [cat1, cat2, cat3], [cat1, cat2, cat3]]
        # response : [res1, res2, res3, res4, res5, res6, res7, res8, res9]
        # stream_ids : [stream_id1, stream_id2, stream_id3]
        i = 0
        for response, categories, stream_id in zip(responses, categories_per_stream, stream_ids):
            for cat in categories:
                # res = responses[i]
                i += 1
                predict = self.parse_prediction(response)
                print_only_debug_mode(
                    f"stream_id : {stream_id}\tcategory : {cat}\tpredict : {predict}"
                )
                if predict in MODEL_OUTPUTS[0] and cat in FALLDOWN_CATEGORY:  # falldown
                    self.duration_queue[stream_id][cat].append(1)
                elif predict in MODEL_OUTPUTS[1] and cat in FIRE_CATEGORY:  # fire
                    self.duration_queue[stream_id][cat].append(1)
                else:
                    self.duration_queue[stream_id][cat].append(0)
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
    def parse_prediction(pred_str: str) -> str:
        if not isinstance(pred_str, str):
            return 'parsing_failed'
        try:
            clean_str = pred_str
            if '```json' in clean_str:
                clean_str = clean_str.split('```json')[1].split('```')[0]
            elif '```' in clean_str:
                clean_str = clean_str.split('```')[1].split('```')[0]
            clean_str = clean_str.strip()
            start_brace, end_brace = clean_str.find('{'), clean_str.rfind('}')
            if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
                json_part = clean_str[start_brace : end_brace + 1]
                try:
                    data = json.loads(json_part)
                    category = data.get('category')
                    if isinstance(category, str) and category.lower() in ['falldown', 'normal']:
                        return category.lower()
                except json.JSONDecodeError:
                    pass
            cat_match = re.search(
                r'["\']category["\']\s*:\s*["\'](falldown|normal)["\']', clean_str, re.IGNORECASE
            )
            if cat_match:
                return cat_match.group(1).lower()
            return 'no_json_found'
        except Exception:
            return 'parsing_failed'
