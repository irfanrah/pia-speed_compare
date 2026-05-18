from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
import torch

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.global_config import RET_EVENT_KEY, USER_PARAM_KEY
from pia_prod.AI.modules.pe_vle_2stage_async.config import (
    ALARM_DURATION_THRESHOLD,
    CATEGORY_EVENT_MAP,
    DEVICE,
    INDEX_MAPPING,
    QUEUE_SIZE,
    TOP_CANDIDATE,
    TOP_K,
)


class PeVleEventManager(EventBase):
    def __init__(self):
        super().__init__(
            alarm_duration=ALARM_DURATION_THRESHOLD
        )
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=(QUEUE_SIZE))))
        self.event_status = defaultdict(lambda: defaultdict(int))
        self.category_count = len(INDEX_MAPPING)

    def prepare_vectors_on_gpu(self, category_txt_vectors: dict):
        self.gpu_ids = torch.from_numpy(category_txt_vectors["ids"]).to(DEVICE)
        self.gpu_class = torch.from_numpy(category_txt_vectors["class_list"]).to(DEVICE)
        self.gpu_vectors = category_txt_vectors["vectors"].to(DEVICE)
        self.prompt_list_np = category_txt_vectors["prompt_list"]

    def _calc_topK(self, top_indices, abnormal_last_idx):
        if (top_indices < abnormal_last_idx).sum() >= TOP_K:
            return True
        return False

    def _decide_top_category_opt(self, sim: torch.Tensor):
        N = sim.numel()
        target_count = TOP_CANDIDATE
        search_k = min(N, target_count * 10)

        top_scores, top_indices = torch.topk(sim, k=search_k, sorted=True)

        cand_ids = self.gpu_ids[top_indices]
        cand_classes = self.gpu_class[top_indices]

        cand_ids_cpu = cand_ids.tolist()

        selected_indices = []
        seen_ids = set()

        for idx, cid in enumerate(cand_ids_cpu):
            if cid == 0:
                selected_indices.append(idx)
            elif cid not in seen_ids:
                seen_ids.add(cid)
                selected_indices.append(idx)

            if len(selected_indices) >= target_count:
                break

        if not selected_indices:
            return np.array([]), (np.array([]), np.array([]), np.array([]))

        sel_idxs_tensor = torch.as_tensor(selected_indices, device=sim.device)

        final_sim = top_scores[sel_idxs_tensor]
        final_class = cand_classes[sel_idxs_tensor]

        final_real_indices = top_indices[sel_idxs_tensor].cpu().numpy()
        final_prompt = self.prompt_list_np[final_real_indices]

        counts = torch.bincount(final_class, minlength=self.category_count)

        winners = torch.where(counts == counts.max())[0]

        return winners.cpu().numpy(), (
            final_sim.cpu().numpy(),
            final_class.cpu().numpy(),
            final_prompt,
        )

    def process_event(self, category_id, predict, stream_id):
        for event, category_set in CATEGORY_EVENT_MAP.items():
            if category_id in category_set:
                value = 1 if predict == event else 0
                self.duration_queue[stream_id][category_id].append(value)
                break
        else:
            self.duration_queue[stream_id][category_id].append(0)

    def process_category(self, user_param, predicts, stream_id):
        predicts = [CATEGORY_EVENT_MAP[INDEX_MAPPING[p]] for p in predicts.tolist() if p != 0]
        predicts = [item for sublist in predicts for item in sublist]

        for category_id in user_param[USER_PARAM_KEY][RET_EVENT_KEY]:
            if category_id in predicts:
                self.duration_queue[stream_id][category_id].append(1)
            else:
                self.duration_queue[stream_id][category_id].append(0)

    def update(
        self,
        vis_vectors: Dict[str, deque],
        stream_ids: List,
        user_params: List,
    ):
        predict_infos = []
        for stream_id, user_param in zip(stream_ids, user_params):
            meanpooling_visual_vector = (sum(vis_vectors[stream_id]) / len(vis_vectors[stream_id]))[
                None, ::
            ]
            sim = (meanpooling_visual_vector @ self.gpu_vectors.T).squeeze(dim=0)
            predicts, predict_info = self._decide_top_category_opt(sim)
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
                    # Composite key — multiple categories firing on one stream
                    # don't collide. Empty value-side category_id so
                    # ServiceBase.get_alarm_with_uuid doesn't double-prefix.
                    alarms[f"{stream_id}__{category_id}"] = [
                        self.EVENT_STATUS_DICT[now_status],
                        "",
                    ]
        return alarms
