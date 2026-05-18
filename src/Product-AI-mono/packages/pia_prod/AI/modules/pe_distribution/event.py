from pia_prod.AI.bases.event_base import EventBase
from collections import deque, defaultdict
from typing import Dict, List
import torch
import numpy as np
from pia_prod.AI.modules.pe_distribution.config import (
    QUEUE_SIZE,
    ALARM_DURATION_THRESHOLD,
    INDEX_MAPPING,
    CATEGORY_EVENT_MAP,
    DEVICE,
    IOU_THRESHOLD,
    IOU_HIST_BINS,
    NORMAL_CATEGORY_ID,
)
from pia_prod.AI.global_config import USER_PARAM_KEY, RET_EVENT_KEY


class PEDistributionEventManager(EventBase):
    def __init__(self):
        super().__init__(
            # queue_size=QUEUE_SIZE,
            alarm_duration=ALARM_DURATION_THRESHOLD
        )
        self.queue_size = QUEUE_SIZE
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=(self.queue_size))))
        self.event_status = defaultdict(lambda: defaultdict(int))
        self.category_count = len(INDEX_MAPPING)

    def prepare_vectors_on_gpu(self, category_txt_vectors: dict):
        self.gpu_ids = torch.from_numpy(category_txt_vectors["ids"]).to(DEVICE)
        self.gpu_class = torch.from_numpy(category_txt_vectors["class_list"]).to(DEVICE)
        self.gpu_vectors = category_txt_vectors["vectors"].to(DEVICE)
        self.prompt_list_np = category_txt_vectors["prompt_list"]
        # 카테고리별 점수 마스크 사전 계산 (매 프레임 재계산 회피)
        self.class_list_np = np.asarray(category_txt_vectors["class_list"])
        self.category_masks = {
            cid: self.class_list_np == cid for cid in INDEX_MAPPING.keys()
        }

    @staticmethod
    def _compute_hist_iou(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """동일한 bin edges 를 가진 두 정규화 히스토그램의 면적 기반 IoU."""
        if hist_a is None or hist_b is None:
            return 0.0
        inter = float(np.minimum(hist_a, hist_b).sum())
        union = float(np.maximum(hist_a, hist_b).sum())
        return (inter / union) if union > 0 else 0.0

    def _decide_categories_by_distribution(self, sim: torch.Tensor):
        """
        카테고리별 유사도 분포 히스토그램을 만들고,
        normal(=NORMAL_CATEGORY_ID) 카테고리와의 hist-IoU < IOU_THRESHOLD 인 이벤트 카테고리를 후보로 선정.
        """
        scores_np = sim.detach().cpu().numpy()

        # 카테고리별 점수 분리
        scores_by_category = {}
        for cat_id, mask in self.category_masks.items():
            s = scores_np[mask]
            if len(s) > 0:
                scores_by_category[cat_id] = s

        if not scores_by_category:
            return np.array([], dtype=int), {}

        # 전체 점수에서 공통 bin edges 결정
        flat = np.concatenate(list(scores_by_category.values()))
        x_lo = float(flat.min())
        x_hi = float(flat.max())
        if x_hi - x_lo < 1e-6:
            x_hi = x_lo + 1e-6
        edges = np.linspace(x_lo, x_hi, IOU_HIST_BINS + 1)

        # 카테고리별 정규화 히스토그램 (sum -> 1)
        hist_for_iou = {}
        for cid, s in scores_by_category.items():
            c, _ = np.histogram(s, bins=edges)
            tot = c.sum()
            hist_for_iou[cid] = (c / tot) if tot > 0 else c.astype(float)

        # normal vs 이벤트 카테고리 IoU
        ious_per_cat = {}
        if NORMAL_CATEGORY_ID in hist_for_iou:
            for eid in hist_for_iou:
                if eid == NORMAL_CATEGORY_ID:
                    continue
                ious_per_cat[eid] = self._compute_hist_iou(
                    hist_for_iou.get(NORMAL_CATEGORY_ID),
                    hist_for_iou.get(eid),
                )

        # IoU < threshold 인 이벤트 카테고리만 predicts
        predicts = np.array(
            [eid for eid, iv in ious_per_cat.items() if iv < IOU_THRESHOLD],
            dtype=int,
        )
        return predicts, ious_per_cat

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
            predicts, ious_per_cat = self._decide_categories_by_distribution(sim)
            predict_infos.append(ious_per_cat)
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
                    alarms[stream_id] = [self.EVENT_STATUS_DICT[now_status], category_id]
        return alarms
