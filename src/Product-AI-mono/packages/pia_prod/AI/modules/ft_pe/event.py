from collections import defaultdict, deque
from typing import Dict, List

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.global_config import RET_EVENT_KEY, USER_PARAM_KEY
from pia_prod.AI.modules.ft_pe.config import CATEGORY_EVENT_MAP


class FTPEEventManager(EventBase):
    """
    FT_PE alarm queue mechanism (per stream × per category).

    - pe_violence와 동일한 queue_size / threshold 기반 smoothing 사용
    - perception_encoder처럼 카테고리별 독립 큐 유지
    - user_param.retEvent에 등록된 카테고리만 업데이트/알람 판정
    """

    def __init__(self, alarm_queue_size: int, alarm_threshold: int):
        super().__init__(alarm_duration=alarm_threshold)
        self.alarm_queue_size = alarm_queue_size
        self.alarm_threshold = alarm_threshold
        self.duration_queue = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.alarm_queue_size))
        )
        self.event_status = defaultdict(lambda: defaultdict(int))

    def update(
        self,
        predicts: List[Dict[str, bool]],
        stream_ids: List[str],
        user_params: List,
    ):
        for stream_id, user_param, pred in zip(stream_ids, user_params, predicts):
            requested_events = user_param[USER_PARAM_KEY][RET_EVENT_KEY]
            for ret_key in requested_events:
                is_abnormal = 0
                for class_name, category_set in CATEGORY_EVENT_MAP.items():
                    if ret_key in category_set:
                        is_abnormal = int(bool(pred.get(class_name, False)))
                        break
                self.duration_queue[stream_id][ret_key].append(is_abnormal)
        return self.check_alarm_duration(stream_ids)

    def check_alarm_duration(self, stream_ids):
        alarms = {}
        for stream_id in stream_ids:
            cat_dict = self.duration_queue.get(stream_id)
            if cat_dict is None:
                continue
            for category_id, queue in cat_dict.items():
                before_status = self.event_status[stream_id][category_id]
                is_over_queue = int(sum(queue) >= self.alarm_threshold)
                now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
                self.event_status[stream_id][category_id] = now_status
                if now_status in [1, 3]:
                    # Composite key prevents multi-category overwrite on the same stream.
                    # category_id cleared in value so ServiceBase.get_alarm_with_uuid does not re-append it.
                    alarms[f"{stream_id}__{category_id}"] = [
                        self.EVENT_STATUS_DICT[now_status],
                        "",
                    ]
        return alarms
