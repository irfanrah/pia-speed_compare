from queue import Queue
from collections import defaultdict, deque
import json
import torch
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.qwen3_vl_embedding.event import Qwen3VLEEventManager
from pia_prod.AI.modules.qwen3_vl_embedding.config import (
    ALL_CATEGORIES,
    CATEGORY_EVENT_MAP,
    DEVICE,
    IMG_SIZE,
    QWEN3VLE_ID,
    QWEN3VLE_MODEL_HF_PATH,
    QWEN3VLE_TEXT_FEATURES_PATH,
    TEMPORAL_SIZE,
)
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.ai.model import PiaTorchModel
from pia_prod.AI.modules.qwen3_vl_embedding.roi_manager import Qwen3VLERoIManager
from pia_prod.AI.utils.init import logger
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    RET_EVENT_KEY,
    STREAM_IDS_KEY,
    USER_PARAM_KEY,
    USER_PARAMS_KEY,
)
from typing import Dict, List


class Qwen3VLEService(ServiceBase):
    """
    Qwen3-VL-Embedding service (torch / HF variant).

    Pipeline:
        1. ROI-crop incoming frames.
        2. Resize + buffer until TEMPORAL_SIZE is reached.
        3. Stack into a video tensor and run the HF model → video embedding.
        4. Compare each embedding against normal / target text embeddings
           per category; flag each requested category independently as
           abnormal/normal (multi-category — a single scene may trigger
           multiple categories, e.g. fire + smoke).
        5. Feed predictions into the event manager; fetch alarms.
    """
    def __init__(self, analysis_data_queue: Queue):
        self.img_size = IMG_SIZE
        self.temporal_size = TEMPORAL_SIZE
        self.frame_buffers = defaultdict(lambda: deque(maxlen=self.temporal_size))
        self.is_needed_cvt_color = True
        self.debug = False

        super().__init__(analysis_data_queue)

    def _init_values(self):
        self.category_to_txt_embeddings = {}

    def _load_model(self):
        self.config = T2VRetConfig(
            model_path=QWEN3VLE_MODEL_HF_PATH,
            device=DEVICE,
            tile_config=None,
            model_name=QWEN3VLE_ID,
            temporal_size=self.temporal_size,
            img_size=self.img_size,
        )
        self.model = PiaTorchModel(
            target_task="RET",
            target_model="Qwen3VLEmbedding",
            config=self.config,
        )
        self._load_text_embeddings()

    def _load_roi_manager(self):
        return Qwen3VLERoIManager()

    def _load_event_manager(self):
        return Qwen3VLEEventManager()


    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def _load_text_embeddings(self):
        with open(QWEN3VLE_TEXT_FEATURES_PATH) as f:
            data = json.load(f)

        self.category_to_txt_embeddings = {}
        for class_name, block in data.items():
            feats = block.get("text_features", {})
            normal, target = feats.get("normal", []), feats.get(class_name, [])
            if not normal or not target: continue
            
            n_mat = F.normalize(torch.tensor(normal, dtype=torch.float32, device=DEVICE), p=2, dim=-1)
            t_mat = F.normalize(torch.tensor(target, dtype=torch.float32, device=DEVICE), p=2, dim=-1)
            
            self.category_to_txt_embeddings[class_name] = {
                "normal": n_mat.t().contiguous(),
                "target": t_mat.t().contiguous(),
            }

        assert self.category_to_txt_embeddings, (
            f"No text_features loaded from {QWEN3VLE_TEXT_FEATURES_PATH}"
        )

    def _extract_ready_videos(self, stream_ids, original_batches, user_params):
        """
        Checks buffers for completed temporal windows. 
        Returns stacked video tensors and corresponding metadata if ready, else None.
        """
        ready_stream_ids = []
        video_tensors = []
        ready_user_params = []
        latest_frames = {}

        user_param_map = {sid: param for sid, param in zip(stream_ids, user_params)}
        original_frame_map = {sid: frame for sid, frame in zip(stream_ids, original_batches)}

        for stream_id in stream_ids:
            buffer = self.frame_buffers[stream_id]
            
            if len(buffer) == self.temporal_size:
                video_tensors.append(torch.stack(list(buffer)))
                ready_stream_ids.append(stream_id)
                ready_user_params.append(user_param_map[stream_id])
                latest_frames[stream_id] = original_frame_map[stream_id] 
                
                buffer.clear()

        if not ready_stream_ids:
            return None

        batched_videos = torch.stack(video_tensors)  # [B, T, C, H, W]
        return batched_videos, ready_stream_ids, ready_user_params, latest_frames

    def _get_category_predictions(
        self, vid_embeddings, user_params,
    ) -> List[Dict[str, bool]]:
        """
        Return one dict per video with each category's independent verdict.

        Decision rule: for each requested category, flag abnormal iff max sim
        to its 'target' prompts > max sim to its own 'normal' prompts. A scene
        can flag multiple categories at once (e.g. fire + smoke).
        """
        video_batch_size = vid_embeddings.size(0)
        categories = list(CATEGORY_EVENT_MAP.keys())

        vid_embeddings = F.normalize(vid_embeddings, p=2, dim=1).float()

        # Compute each category's abnormal verdict across the whole batch
        cat_verdicts: Dict[str, List[bool]] = {}
        for category in categories:
            embeds = self.category_to_txt_embeddings.get(category)
            if embeds is None:
                cat_verdicts[category] = [False] * video_batch_size
                continue

            normal_max_sim = (vid_embeddings @ embeds["normal"]).max(dim=1).values
            target_max_sim = (vid_embeddings @ embeds["target"]).max(dim=1).values
            cat_verdicts[category] = (target_max_sim > normal_max_sim).detach().cpu().tolist()

        final_predictions: List[Dict[str, bool]] = []
        for batch_idx in range(video_batch_size):
            user_requested_ids = set(
                user_params[batch_idx].get(USER_PARAM_KEY, {}).get(RET_EVENT_KEY, [])
            )

            unsupported = user_requested_ids - set(ALL_CATEGORIES)
            if unsupported:
                logger.warning(
                    f"[Qwen3VLE] Unsupported retEvent categories received. retEvent: {unsupported}"
                )

            # Only surface verdicts for categories the user actually requested;
            # others stay False so the event manager never accumulates for them.
            per_cat: Dict[str, bool] = {}
            for category in categories:
                category_ids = set(CATEGORY_EVENT_MAP.get(category, []))
                if user_requested_ids.intersection(category_ids):
                    per_cat[category] = bool(cat_verdicts[category][batch_idx])
                else:
                    per_cat[category] = False

            final_predictions.append(per_cat)

        return final_predictions

    def _predict(self, batches, stream_ids, user_params):
        """Preprocessing + model inference. Returns raw predictions without event manager."""
        # 1. Preprocess frames (crop, and append to temporal buffer)   
        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)

        for stream_id, batch in zip(stream_ids, cropped_batches):
            resized_batch = TF.resize(batch, self.img_size, antialias=True)
            self.frame_buffers[stream_id].append(resized_batch)

        # 2. Extract videos that have reached TEMPORAL_SIZE
        ready_data = self._extract_ready_videos(stream_ids, batches, user_params)
        if not ready_data:
            return None
        
        batched_videos, ready_stream_ids, ready_user_params, latest_frames = ready_data

        # 3. Model Inference
        with torch.inference_mode():
            vid_embeddings = self.model(video=batched_videos, text=None)

        # 4. Calculate similarity and filter out unrequested categories
        predictions = self._get_category_predictions(vid_embeddings, ready_user_params)

        return {
            "predictions": predictions,
            STREAM_IDS_KEY: ready_stream_ids,
            USER_PARAMS_KEY: ready_user_params,
            "latest_frames": latest_frames,
        }

    # -------------------------------------------------------------------------
    # Main Detect Function
    # -------------------------------------------------------------------------
    def _detect(self, **datas):
        batches = datas[BATCHES_KEY]
        stream_ids = datas[STREAM_IDS_KEY]
        user_params = datas[USER_PARAMS_KEY]

        if not self.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)

        result = self._predict(batches, stream_ids, user_params)
        if not result:
            return None

        # Update Event Manager & Fetch Alarms
        alarms = self.alarm_event_manager.update(
            result["predictions"], result[STREAM_IDS_KEY], result[USER_PARAMS_KEY]
        )
        if not alarms:
            return None

        if self.debug:
            self.send_alarm({
                ALARMS_KEY: alarms,
                BATCHES_KEY: datas[BATCHES_KEY],
                STREAM_IDS_KEY: datas[STREAM_IDS_KEY],
                USER_PARAMS_KEY: datas[USER_PARAMS_KEY],
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            })

        frame_list = [result["latest_frames"][sid] for sid in result[STREAM_IDS_KEY]]
        return {
            ALARMS_KEY: alarms,
            BATCHES_KEY: frame_list,
            STREAM_IDS_KEY: result[STREAM_IDS_KEY],
            USER_PARAMS_KEY: result[USER_PARAMS_KEY],
            IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
        }