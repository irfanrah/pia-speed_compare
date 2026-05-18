from pia_prod.AI.bases.event_base import EventBase
from collections import deque, defaultdict
from typing import Dict, List
import torch
import numpy as np
from pia_prod.AI.modules.pe_binary.config import (
    QUEUE_SIZE,
    TOP_CANDIDATE,
    TOP_K,
    ALARM_DURATION_THRESHOLD,
    INDEX_MAPPING,
    CATEGORY_EVENT_MAP,
    DEVICE,
)
from pia_prod.AI.global_config import USER_PARAM_KEY, RET_EVENT_KEY


class PEBinaryEventManager(EventBase):
    def __init__(self):
        super().__init__(
            # queue_size=QUEUE_SIZE,
            alarm_duration=ALARM_DURATION_THRESHOLD
        )
        self.queue_size = QUEUE_SIZE
        self.duration_queue = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=(self.queue_size)))
        )
        self.event_status = defaultdict(lambda: defaultdict(int))
        self.category_count = len(INDEX_MAPPING)

    def prepare_vectors_on_gpu(self, category_txt_vectors: dict):
        self.gpu_ids = torch.from_numpy(category_txt_vectors["ids"]).to(DEVICE)
        self.gpu_class = torch.from_numpy(category_txt_vectors["class_list"]).to(DEVICE)
        self.gpu_vectors = category_txt_vectors["vectors"].to(DEVICE)
        self.prompt_list_np = category_txt_vectors["prompt_list"]

        self.class_idx_map = {
            cls: torch.nonzero(self.gpu_class == cls, as_tuple=True)[0] for cls in INDEX_MAPPING
        }

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

    def _decide_top_category_opt_binary(self, sim: torch.Tensor):
        target_count = TOP_CANDIDATE
        categories = [c for c in INDEX_MAPPING if c != 0]
        search_k = target_count * 2

        cls_scores, cls_indices = [], []
        for cls, cls_idx in self.class_idx_map.items():
            if cls_idx.numel() == 0:
                continue
            k = min(cls_idx.numel(), search_k)
            scores, local = torch.topk(sim[cls_idx], k=k, sorted=False)
            cls_scores.append(scores)
            cls_indices.append(cls_idx[local])

        if not cls_scores:
            return np.array([]), (np.array([]), np.array([]), np.array([]))

        top_scores = torch.cat(cls_scores)
        top_indices = torch.cat(cls_indices)
        top_scores, sort_order = torch.sort(top_scores, descending=True)
        top_indices = top_indices[sort_order]

        cand_ids = self.gpu_ids[top_indices]
        cand_classes = self.gpu_class[top_indices]

        cand_ids_cpu = cand_ids.tolist()
        cand_classes_cpu = cand_classes.tolist()

        selected_per_cat = {cat: [] for cat in categories}
        seen_ids_per_cat = {cat: set() for cat in categories}
        unfilled = set(categories)

        for pos in range(len(cand_ids_cpu)):
            if not unfilled:
                break
            cls = cand_classes_cpu[pos]
            cur_id = cand_ids_cpu[pos]

            if cls == 0:
                target_cats = list(unfilled)
            elif cls in unfilled:
                target_cats = (cls,)
            else:
                continue

            for cat in target_cats:
                if cur_id != 0:
                    seen = seen_ids_per_cat[cat]
                    if cur_id in seen:
                        continue
                    seen.add(cur_id)
                bucket = selected_per_cat[cat]
                bucket.append(pos)
                if len(bucket) >= target_count:
                    unfilled.discard(cat)

        all_winners = set()
        sims, classes, prompts = [], [], []

        for cat in categories:
            sel = selected_per_cat[cat]
            if not sel:
                continue

            sel_idxs_tensor = torch.as_tensor(sel, device=sim.device)

            final_sim = top_scores[sel_idxs_tensor]
            final_class = cand_classes[sel_idxs_tensor]
            final_real_indices = top_indices[sel_idxs_tensor].cpu().numpy()
            final_prompt = self.prompt_list_np[final_real_indices]

            counts = torch.bincount(final_class, minlength=self.category_count)
            winners = torch.where(counts == counts.max())[0]

            all_winners.update(winners.cpu().numpy().tolist())
            sims.append(final_sim.cpu().numpy())
            classes.append(final_class.cpu().numpy())
            prompts.append(final_prompt)

        if not all_winners:
            return np.array([]), (np.array([]), np.array([]), np.array([]))

        return (
            np.array(sorted(all_winners)),
            (np.concatenate(sims), np.concatenate(classes), np.concatenate(prompts)),
        )

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
            predicts, predict_info = self._decide_top_category_opt_binary(sim)
            predict_infos.append(predict_info)
            self.process_category(user_param, predicts, stream_id)
        return self.check_alarm_duration(), predict_infos

    def check_alarm_duration(self):
        alarms = {}
        for stream_id, cat_dict in self.duration_queue.items():
            for category_id, value in cat_dict.items():
                before_status = self.event_status[stream_id][category_id]
                is_over_queue = int(sum(value) >= self.alarm_duration)
                now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
                self.event_status[stream_id][category_id] = now_status
                if now_status in [1, 3]:
                    # ✅ 카테고리별로 독립된 key 사용
                    key = f"{stream_id}__{category_id}"
                    alarms[key] = [self.EVENT_STATUS_DICT[now_status], None]
        return alarms
