"""
Qwen3VLE Sync Service — vLLM in-process variant.

Mirrors `qwen3vle.service.Qwen3VLEService` exactly except for the embedding
step: instead of POST-ing JPEG-encoded frames to an external embedding
server, this version loads vLLM inside the same Python process and calls
`llm.embed(prompts)` directly. That eliminates:

  * Client-side JPEG encode + base64 encode (~5 ms/frame at 768×768 px).
  * HTTP serialize + transport + deserialize.
  * Server-side base64 decode + PIL decode (~5–10 ms/frame).
  * Server-side asyncio orchestration (thread-pool queue, semaphore, gather).
  * JSON response parse on the client.

The trade-off is that the heavy vLLM model lives in the same Python
interpreter as the rest of pia_prod, so GPU memory has to be co-budgeted
with PE / ft_pe / other in-process models. Use this only when you're
willing to trade memory headroom for the latency win.

Wire contract is identical to `Qwen3VLEService` — same `_detect(...)`
signature, same `predictions` dict shape, same event-manager interface —
so it can be slotted in anywhere Qwen3VLEService is currently used.
"""

import json
from collections import defaultdict, deque
from queue import Queue
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.qwen3vle_sync.config import (
    ALL_CATEGORIES,
    CATEGORY_EVENT_MAP,
    DEFAULT_INSTRUCTION,
    DEVICE,
    IMG_SIZE,
    QWEN3VLE_SYNC_DTYPE,
    QWEN3VLE_SYNC_ENFORCE_EAGER,
    QWEN3VLE_SYNC_GPU_MEMORY_UTILIZATION,
    QWEN3VLE_SYNC_MAX_MODEL_LEN,
    QWEN3VLE_SYNC_MODEL_PATH,
    QWEN3VLE_SYNC_TEXT_FEATURES_PATH,
    TEMPORAL_FACTOR,
    TEMPORAL_SIZE,
)
from pia_prod.AI.modules.qwen3vle_sync.event import Qwen3VLESyncEventManager
from pia_prod.AI.modules.qwen3vle_sync.roi_manager import Qwen3VLESyncRoIManager
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


class Qwen3VLESyncService(ServiceBase):
    """
    Qwen3-VL-Embedding service (vLLM in-process variant).

    Pipeline (identical to the remote `Qwen3VLEService`, only the embed
    step differs):
        1. ROI-crop incoming frames.
        2. Resize + buffer until TEMPORAL_SIZE is reached.
        3. Call vLLM `llm.embed(...)` directly on the batched videos.
        4. Compare each embedding against normal / target text embeddings
           per category; flag each requested category independently.
        5. Feed predictions into the event manager; fetch alarms.
    """

    def __init__(self, analysis_data_queue: Queue):
        self.img_size = IMG_SIZE
        self.temporal_size = TEMPORAL_SIZE
        self.temporal_factor = TEMPORAL_FACTOR
        self.frame_buffers = defaultdict(lambda: deque(maxlen=self.temporal_size))
        self.is_needed_cvt_color = True
        self.debug = False

        super().__init__(analysis_data_queue)

    def _init_values(self):
        self.category_to_txt_embeddings = {}

    def _load_model(self):
        """Load vLLM in-process and cache the tokenizer + pooling params.

        Heavy operation — pulls vLLM, allocates the GPU, captures CUDA
        graphs. Called once at service construction. Same engine kwargs
        the embedding_server uses, so per-clip GPU cost is identical;
        we just skip the HTTP wrapper.
        """
        from vllm import LLM
        from vllm.pooling_params import PoolingParams

        logger.info(
            f"[Qwen3VLESync] Loading vLLM in-process from {QWEN3VLE_SYNC_MODEL_PATH} "
            f"(dtype={QWEN3VLE_SYNC_DTYPE}, gpu_mem_util={QWEN3VLE_SYNC_GPU_MEMORY_UTILIZATION})"
        )
        self._llm = LLM(
            model=QWEN3VLE_SYNC_MODEL_PATH,
            runner="pooling",
            trust_remote_code=True,
            dtype=QWEN3VLE_SYNC_DTYPE,
            max_model_len=QWEN3VLE_SYNC_MAX_MODEL_LEN,
            gpu_memory_utilization=QWEN3VLE_SYNC_GPU_MEMORY_UTILIZATION,
            limit_mm_per_prompt={"video": 1, "image": 1},
            enforce_eager=QWEN3VLE_SYNC_ENFORCE_EAGER,
        )
        self._pooling_params = PoolingParams(task="embed")
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("[Qwen3VLESync] vLLM ready")

        self._load_text_embeddings()

    def _load_roi_manager(self):
        return Qwen3VLESyncRoIManager()

    def _load_event_manager(self):
        return Qwen3VLESyncEventManager()

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def _load_text_embeddings(self):
        """Same text-feature loader as the remote service — text features
        are tiny and live alongside the model checkpoint."""
        with open(QWEN3VLE_SYNC_TEXT_FEATURES_PATH) as f:
            data = json.load(f)

        self.category_to_txt_embeddings = {}
        for class_name, block in data.items():
            feats = block.get("text_features", {})
            normal, target = feats.get("normal", []), feats.get(class_name, [])
            if not normal or not target:
                continue

            n_mat = F.normalize(
                torch.tensor(normal, dtype=torch.float32, device=DEVICE), p=2, dim=-1
            )
            t_mat = F.normalize(
                torch.tensor(target, dtype=torch.float32, device=DEVICE), p=2, dim=-1
            )

            self.category_to_txt_embeddings[class_name] = {
                "normal": n_mat.t().contiguous(),
                "target": t_mat.t().contiguous(),
            }

        assert self.category_to_txt_embeddings, (
            f"No text_features loaded from {QWEN3VLE_SYNC_TEXT_FEATURES_PATH}"
        )

    def _build_video_input(self, video_tensor: torch.Tensor):
        """Convert one [T, C, H, W] resized RGB torch tensor into the
        (np.ndarray, metadata_dict) tuple vLLM's Qwen3-VL data parser
        requires for `multi_modal_data["video"]`. Pads T<TEMPORAL_FACTOR
        by repeating the last frame — same logic as the embedding_server.
        """
        # vLLM wants HWC uint8 — convert from CHW torch (possibly float
        # after TF.resize antialiasing) by clamping + casting.
        arr = video_tensor.detach()
        if arr.is_floating_point():
            arr = arr.clamp(0, 255).to(torch.uint8)
        # [T, C, H, W] -> [T, H, W, C]
        arr_np = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        T = arr_np.shape[0]
        if T < self.temporal_factor:
            pad = np.repeat(arr_np[-1:], self.temporal_factor - T, axis=0)
            arr_np = np.concatenate([arr_np, pad], axis=0)
            T = arr_np.shape[0]

        # Metadata matches embedding_server's _build_video_metadata so the
        # text-feature anchors stay calibrated across the two services.
        metadata = {
            "fps": 2.0,
            "duration": T / 2.0,
            "total_num_frames": T,
            "frames_indices": list(range(T)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        }
        return arr_np, metadata

    def _build_prompts(self, batched_videos: torch.Tensor) -> List[dict]:
        """Build one vLLM prompt per video in the batch. Each prompt
        carries the chat-template-rendered string plus the multi_modal_data
        tuple. Same rendering as the embedding_server uses, so embeddings
        are byte-identical to the HTTP path for the same input."""
        prompts = []
        for video in batched_videos:  # [T, C, H, W] per element
            video_np, metadata = self._build_video_input(video)
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
                {"role": "user", "content": [{"type": "video", "video": video_np}]},
            ]
            prompt_text = self._tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            prompts.append({
                "prompt": prompt_text,
                "multi_modal_data": {"video": (video_np, metadata)},
            })
        return prompts

    def _embed_via_vllm(self, batched_videos: torch.Tensor) -> torch.Tensor:
        """In-process replacement for the remote service's
        `_embed_via_server`. Runs one batched `llm.embed(...)` call and
        returns [B, D] float32 on DEVICE.

        Returns shape [B, D] where D is the embedding dimension (model-
        dependent; 2048 for Qwen3-VL-Embedding-2B). All embeddings are
        unnormalized — `_get_category_predictions` does the L2 normalize.
        """
        prompts = self._build_prompts(batched_videos)
        outs = self._llm.embed(prompts, pooling_params=self._pooling_params)

        embeddings = []
        for out in outs:
            pooling_out = out.outputs
            if isinstance(pooling_out, list):
                pooling_out = pooling_out[0]
            # Some vLLM versions expose `.embedding`, others `.data`.
            vec = pooling_out.embedding if hasattr(pooling_out, "embedding") else pooling_out.data
            embeddings.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
        return torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)

    def _extract_ready_videos(self, stream_ids, original_batches, user_params):
        """Identical to the remote service. Pulls completed temporal
        windows out of `self.frame_buffers` and returns the stacked
        [B, T, C, H, W] tensor plus per-stream metadata."""
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
        """Identical to the remote service. For each requested category,
        flag abnormal iff max sim to its 'target' prompts > max sim to its
        own 'normal' prompts."""
        video_batch_size = vid_embeddings.size(0)
        categories = list(CATEGORY_EVENT_MAP.keys())

        vid_embeddings = F.normalize(vid_embeddings, p=2, dim=1).float()

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
                    f"[Qwen3VLESync] Unsupported retEvent categories received. retEvent: {unsupported}"
                )

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
        """Preprocessing + in-process embedding. Wire contract identical
        to the remote service so callers (e.g. PeVle2StageSyncService)
        don't need to know which variant they're holding."""
        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)

        for stream_id, batch in zip(stream_ids, cropped_batches):
            resized_batch = TF.resize(batch, self.img_size, antialias=True)
            self.frame_buffers[stream_id].append(resized_batch)

        ready_data = self._extract_ready_videos(stream_ids, batches, user_params)
        if not ready_data:
            return None

        batched_videos, ready_stream_ids, ready_user_params, latest_frames = ready_data

        # The only behavioral difference from the remote service: direct
        # vLLM call instead of an HTTP POST to embedding_server.
        vid_embeddings = self._embed_via_vllm(batched_videos)

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
