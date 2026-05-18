from typing import Dict, List
from collections import deque, defaultdict
from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.qwen3_vl_embedding.config import (
    CATEGORY_EVENT_MAP,
    QWEN3VLE_QUEUE_SIZE,
    QWEN3VLE_ALARM_DURATION_THRESHOLD,
)
from pia_prod.AI.global_config import USER_PARAM_KEY, RET_EVENT_KEY


class Qwen3VLEEventManager(EventBase):
    """
    Multi-category Event Manager.

    - The service emits one dict per stream mapping each class_name
      (CATEGORY_EVENT_MAP key, e.g. "fire") to an independent abnormal bool.
      Multiple classes can be True simultaneously.
    - Per-stream x per-category duration queue smooths the verdicts; an alarm
      fires for a category once its queue sum crosses the duration threshold.
    - Alarms are keyed by f"{stream_id}__{category_id}" so multiple categories
      triggering on the same stream don't overwrite each other. The value's
      category_id field is intentionally empty — ServiceBase.get_alarm_with_uuid
      uses it to build composite keys and re-appending here would double it.
    """

    def __init__(self):
        super().__init__(alarm_duration=QWEN3VLE_ALARM_DURATION_THRESHOLD)
        self.queue_size = QWEN3VLE_QUEUE_SIZE
        self.duration_queue = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.queue_size))
        )
        self.event_status = defaultdict(lambda: defaultdict(int))

    def update(
        self,
        predicts: List[Dict[str, bool]],
        stream_ids: List[str],
        user_params: List[dict],
    ):
        """Append 1/0 to each requested category's queue based on its per-class verdict."""
        for stream_id, user_param, pred in zip(stream_ids, user_params, predicts):
            requested_events = user_param.get(USER_PARAM_KEY, {}).get(RET_EVENT_KEY, [])

            for ret_key in requested_events:
                is_abnormal = 0
                for class_name, category_set in CATEGORY_EVENT_MAP.items():
                    if ret_key in category_set:
                        is_abnormal = int(bool(pred.get(class_name, False)))
                        break
                self.duration_queue[stream_id][ret_key].append(is_abnormal)

        return self.check_alarm_duration(stream_ids, user_params)

    def check_alarm_duration(
        self, stream_ids: List[str], user_params: List[dict]
    ) -> Dict[str, List]:
        alarms = {}
        for stream_id, user_param in zip(stream_ids, user_params):
            target_categories = user_param.get(USER_PARAM_KEY, {}).get(RET_EVENT_KEY, [])

            for category_id in target_categories:
                before_status = self.event_status[stream_id][category_id]
                now_status = int(
                    sum(self.duration_queue[stream_id][category_id]) >= self.alarm_duration
                )
                final_status = self.STATUS_TRANSITION[before_status][now_status]
                self.event_status[stream_id][category_id] = final_status

                if final_status in [1, 3]:
                    alarms[f"{stream_id}__{category_id}"] = [
                        self.EVENT_STATUS_DICT[final_status],
                        "",
                    ]

        return alarms