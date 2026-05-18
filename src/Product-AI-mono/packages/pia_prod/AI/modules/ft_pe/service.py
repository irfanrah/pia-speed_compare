import json
from collections import defaultdict, deque
from queue import Queue

import torch

from pia.ai.model import PiaONNXTensorRTModel
from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
)
from pia_prod.AI.modules.ft_pe.config import (
    ABNORMAL_CLASS_NAMES,
    ALARM_QUEUE_SIZE,
    ALARM_THRESHOLD,
    DEVICE,
    FT_PE_MODEL_TRT_PATH,
    FT_TEXT_FEATURES_PATH,
    IMG_SIZE,
    NORMAL_CLASS_NAME,
    PREDICTION_SIZE,
    SLIDING_WINDOW_SIZE,
    TEMPORAL_SIZE,
    WINDOW_SIZE,
)
from pia_prod.AI.modules.ft_pe.event import FTPEEventManager
from pia_prod.AI.modules.ft_pe.roi_manager import FTPERoIManager
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image


class FTPEService(ServiceBase):
    """
    Fine-tuned PE multi-category retrieval service.

    - 모델/추론: pe_violence와 동일한 temporal TRT 파이프라인
        · WINDOW_SIZE 프레임을 모아 인코딩 → 비주얼 임베딩 [W, 1024]
        · 인접 윈도우는 SLIDING_WINDOW_SIZE 프레임만큼 겹침 (stride = W − SW)
        · 각 인코드마다 뒤쪽 PREDICTION_SIZE 임베딩만 frame_buffer에 투입
        · TEMPORAL_SIZE 임베딩 버퍼 mean pooling → 비디오 임베딩 [1024]
    - 판정: 카테고리별 Top-1 (max sim_abnormal > max sim_normal)
    - 카테고리: perception_encoder처럼 다중 카테고리 지원
        · CATEGORY_EVENT_MAP을 통해 ret_event 키와 매핑
    """

    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        self.is_needed_cvt_color = True
        self.window_size = WINDOW_SIZE
        self.sliding_window_size = SLIDING_WINDOW_SIZE
        self.prediction_size = PREDICTION_SIZE
        self.stride = max(1, self.window_size - self.sliding_window_size)
        assert 0 <= self.sliding_window_size < self.window_size or (
            self.window_size == 1 and self.sliding_window_size == 0
        ), "sliding_window_size must be in [0, window_size)"
        assert 1 <= self.prediction_size <= self.window_size, (
            "prediction_size must be in [1, window_size]"
        )
        self.frame_buffers = defaultdict(lambda: deque(maxlen=TEMPORAL_SIZE + 1))
        self.gather_frame_buffers = defaultdict(lambda: deque(maxlen=self.window_size))
        self._unconsumed_frames = defaultdict(int)
        self.debug = False

    def _init_values(self):
        # category_txt_vectors[c]: [1024, N] abnormal prompt embeddings for class c
        # category_normal_vectors[c]: [1024, M] normal prompt embeddings SPECIFIC to class c
        # Keeping one normal pool per class — not a union across classes — avoids
        # cross-category normal samples from swallowing the abnormal margin.
        self.category_txt_vectors = {}
        self.category_normal_vectors = {}

    def _load_model(self):
        self.model = PiaONNXTensorRTModel(FT_PE_MODEL_TRT_PATH, device="cuda", half=True)
        self.image_size = IMG_SIZE
        self._load_text_vectors()

    def _load_roi_manager(self):
        return FTPERoIManager()

    def _load_event_manager(self):
        return FTPEEventManager(
            alarm_queue_size=ALARM_QUEUE_SIZE,
            alarm_threshold=ALARM_THRESHOLD,
        )

    def _load_text_vectors(self):
        """
        FT_text_features.json layout:
            {
              "<category>": {
                "text_features": {
                  "normal":    [[1024 floats], ...],
                  "<category>":[[1024 floats], ...],
                },
                ...
              },
              ...
            }

        Each category block keeps its own 'normal' pool. At detection time class c is
        judged as "max sim to c's abnormal > max sim to c's normal", using only c's
        normal samples. Unioning normals across classes let a non-c category's normal
        sample swallow the margin (e.g. a smoke-block normal that's visually close
        to fire content suppressed the fire prediction).

        All tensors are L2-normalized and stored as [1024, N].
        """
        with open(FT_TEXT_FEATURES_PATH) as f:
            data = json.load(f)

        for class_name, block in data.items():
            feats_by_label = block.get("text_features", {})

            normal = feats_by_label.get(NORMAL_CLASS_NAME, [])
            abn = feats_by_label.get(class_name, [])
            if not normal or not abn:
                continue

            n_mat = torch.tensor(normal, dtype=torch.float32, device=DEVICE)  # [M, 1024]
            n_mat = n_mat / n_mat.norm(dim=-1, keepdim=True)
            self.category_normal_vectors[class_name] = n_mat.t().contiguous()  # [1024, M]

            a_mat = torch.tensor(abn, dtype=torch.float32, device=DEVICE)  # [N, 1024]
            a_mat = a_mat / a_mat.norm(dim=-1, keepdim=True)
            self.category_txt_vectors[class_name] = a_mat.t().contiguous()  # [1024, N]

        assert self.category_normal_vectors, (
            f"No per-class '{NORMAL_CLASS_NAME}' text_features found in {FT_TEXT_FEATURES_PATH}"
        )

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        if not self.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)

        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        processed_batches = preprocess_image(
            cropped_batches, size=self.image_size[0], device=DEVICE
        )  # [B, C, H, W]

        for stream_id, batch in zip(stream_ids, processed_batches):
            self.gather_frame_buffers[stream_id].append(batch)
            self._unconsumed_frames[stream_id] += 1

        user_param_map = {sid: param for sid, param in zip(stream_ids, user_params)}

        encode_stream_ids = []
        encode_tensors = []
        for stream_id in stream_ids:
            gbuf = self.gather_frame_buffers[stream_id]
            if len(gbuf) == self.window_size and self._unconsumed_frames[stream_id] >= self.stride:
                encode_tensors.append(torch.stack(list(gbuf)))  # [W, C, H, W]
                self._unconsumed_frames[stream_id] -= self.stride
                encode_stream_ids.append(stream_id)

        if encode_stream_ids:
            encode_input = torch.stack(encode_tensors)  # [B_enc, W, C, H, W]
            with torch.inference_mode():
                embeddings = self.model(encode_input)  # [B_enc, W, 1024]
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            for sid, emb in zip(encode_stream_ids, embeddings):
                # 최근 PREDICTION_SIZE 임베딩만 프레임 버퍼에 투입
                new_embs = emb[-self.prediction_size:]
                self.frame_buffers[sid].extend(new_embs.unbind(0))
                # Drop index 1 to keep the oldest frame for a longer temporal context
                # [0, 1, 2, ..., 8] -> [0, 2, 3, ..., 8]
                if len(self.frame_buffers[sid]) > TEMPORAL_SIZE:
                    del self.frame_buffers[sid][1]

        ready_stream_ids = []
        video_embeddings = []
        for stream_id in stream_ids:
            buf = self.frame_buffers[stream_id]
            if len(buf) >= TEMPORAL_SIZE:
                stacked = torch.stack(list(buf))  # [T, 1024]
                video_embeddings.append(stacked.mean(dim=0))  # [1024]
                ready_stream_ids.append(stream_id)

        if not ready_stream_ids:
            return None

        ready_set = set(ready_stream_ids)
        latest_frames = {sid: f for sid, f in zip(stream_ids, batches) if sid in ready_set}

        vis_vectors = torch.stack(video_embeddings)  # [B_ready, 1024]
        vis_vectors = vis_vectors / vis_vectors.norm(dim=-1, keepdim=True)

        # Per-category decision: sim to class's abnormal prompts vs sim to class's OWN normal prompts.
        category_preds: dict[str, list[bool]] = {}
        for class_name in ABNORMAL_CLASS_NAMES:
            txt_vec = self.category_txt_vectors.get(class_name)
            normal_vec = self.category_normal_vectors.get(class_name)
            if txt_vec is None or normal_vec is None:
                continue
            sim_abn_max = (vis_vectors @ txt_vec).max(dim=1).values
            sim_nrm_max = (vis_vectors @ normal_vec).max(dim=1).values
            category_preds[class_name] = (sim_abn_max > sim_nrm_max).detach().cpu().tolist()

        predicts_per_stream = [
            {cls: category_preds[cls][i] for cls in category_preds}
            for i in range(len(ready_stream_ids))
        ]

        user_param_list = [user_param_map[sid] for sid in ready_stream_ids]
        frame_list = [latest_frames[sid] for sid in ready_stream_ids]

        alarms = self.alarm_event_manager.update(
            predicts_per_stream, ready_stream_ids, user_param_list
        )

        if self.debug:
            debug_results = {
                ALARMS_KEY: alarms,
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
            self.send_alarm(debug_results)

        if len(alarms) > 0:
            return {
                ALARMS_KEY: alarms,
                BATCHES_KEY: frame_list,
                STREAM_IDS_KEY: ready_stream_ids,
                USER_PARAMS_KEY: user_param_list,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
        return None
