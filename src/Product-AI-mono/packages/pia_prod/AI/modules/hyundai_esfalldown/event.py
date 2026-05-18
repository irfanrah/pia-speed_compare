from collections import defaultdict, deque
import torch
import numpy as np
from typing import Dict, List
from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.hyundai_esfalldown.config import (
    ESFALLDOWN_QUEUE_SIZE as QUEUE_SIZE,
    ESFALLDOWN_ALARM_DURATION_THRESHOLD as ALARM_DURATION_THRESHOLD,
    ESFALLDOWN_TOP_CANDIDATE as TOP_CANDIDATE,
    ESFALLDOWN_TOP_K as TOP_K,
    INDEX_MAPPING,
    CATEGORY_EVENT_MAP,
)
from pia_prod.AI.global_config import USER_PARAM_KEY, RET_EVENT_KEY

class ESFalldownEventManager(EventBase):
    def __init__(self):
        super().__init__(
            alarm_duration=ALARM_DURATION_THRESHOLD
        )
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=(QUEUE_SIZE))))
        self.event_status = defaultdict(lambda: defaultdict(int))
        self.category_count = len(INDEX_MAPPING)

    def _calc_topK(self, top_indices, abnormal_last_idx):
        if (top_indices < abnormal_last_idx).sum() >= TOP_K:
            return True
        return False

    def _decide_top_category(self, sim, category_txt_vectors: dict):
        if TOP_CANDIDATE > len(sim):
            candidate = len(sim)
        else:
            candidate = TOP_CANDIDATE
        sorted_sim, indices = torch.sort(sim, descending=True)
        indices = indices.detach().cpu().numpy()
        sorted_ids = category_txt_vectors["ids"][indices]
        sorted_class = category_txt_vectors["class_list"][indices]
        sorted_prompt = category_txt_vectors['prompt_list'][indices]
        predicts = []
        idxs = []
        seen_ids = set()
        for idx, (id, cls) in enumerate(zip(sorted_ids, sorted_class)):
            if id == 0:
                predicts.append(cls)
                idxs.append(idx)
            else:
                if id not in seen_ids:
                    seen_ids.add(id)
                    predicts.append(cls)
                    idxs.append(idx)
            if len(predicts) >= candidate:
                break
        predicts = np.bincount(
            predicts,
            minlength=self.category_count,
        )
        predicts = np.where(predicts == predicts.max())[0]
        predict_info = (sorted_sim[idxs], sorted_class[idxs], sorted_prompt[idxs])
        return predicts, predict_info

    def process_event(self, category_id, predict, stream_id):
        """
        Process the event for a given category_id and prediction.
        Appends 1 or 0 to the duration_queue based on the prediction.
        """
        for event, category_set in CATEGORY_EVENT_MAP.items():
            if category_id in category_set:
                value = 1 if predict == event else 0
                self.duration_queue[stream_id][category_id].append(value)
                break
        else:
            self.duration_queue[stream_id][category_id].append(0)

    def process_category(self, user_param, predicts, stream_id):
        """
        Processes the categories based on user_param and prediction.
        """
        predicts = [CATEGORY_EVENT_MAP[INDEX_MAPPING[p]] for p in predicts if p != 0]
        predicts = [item for sublist in predicts for item in sublist]

        for category_id in user_param[USER_PARAM_KEY][RET_EVENT_KEY]:
            if category_id in predicts:
                self.duration_queue[stream_id][category_id].append(1)
            else:
                self.duration_queue[stream_id][category_id].append(0)

    def update(
        self,
        vis_vectors: Dict[str, deque],
        category_txt_vectors: Dict,
        stream_ids: List,
        user_params: List,
    ):
        predict_infos = []
        for stream_id, user_param in zip(stream_ids, user_params):
            meanpooling_visual_vector = (sum(vis_vectors[stream_id]) / len(vis_vectors[stream_id]))[
                None, ::
            ]
            sim = (meanpooling_visual_vector @ category_txt_vectors['vectors'].T).squeeze(dim=0)
            predicts, predict_info = self._decide_top_category(sim, category_txt_vectors)
            predict_infos.append(predict_info)
            self.process_category(user_param, predicts, stream_id)
        return self.check_alarm_duration(), predict_infos

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
