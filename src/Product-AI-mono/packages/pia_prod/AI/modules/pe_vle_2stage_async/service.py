"""
PE-VLE Service: Stage 1 = PE TRT, Stage 2 = fire-and-forget delegation
to a validation server backed by Qwen3VLE vLLM.

Structure mirrors pe_vqa_2stage line-for-line:
- Stage 1: PE TRT detection (unchanged from PEService).
- Stage 2: build a thumbnail-only payload and POST it fire-and-forget to
  `/api/v1/validate`. The validation server owns the vLLM call, the
  anchor-based classification, and the publish path.

Behavior toggle (PE_VLE_VALIDATION_ENABLED)
-------------------------------------------
    False  → plain PE flow: every alarm publishes directly via match_outputs.
    True   → categories in TWO_STEP_CATEGORIES are delegated to the
             validation server. Other categories publish directly.
"""

import asyncio
import threading
from collections import defaultdict, deque
from queue import Queue
from typing import Optional

import cv2
import base64
import httpx
import torch

from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.DTO.output_handler import match_outputs
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    CV_EVENT_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    RET_EVENT_KEY,
    STREAM_IDS_KEY,
    USER_PARAM_KEY,
    USER_PARAMS_KEY,
    VQA_EVENT_KEY,
)
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image
from pia_prod.AI.modules.pe_vle_2stage_async.config import (
    IMG_SIZE,
    DEVICE,
    TEMPORAL_SIZE,
    PERCEPTION_ENCODER_TXT_FEATURE_PATH,
    PERCEPTION_ENCODER_TRT_PATH,
    VALIDATION_SERVER_ENDPOINT,
    TWO_STEP_CATEGORIES,
    PE_VLE_VALIDATION_ENABLED,
)
from pia_prod.AI.modules.pe_vle_2stage_async.event import PeVleEventManager
from pia_prod.AI.modules.pe_vle_2stage_async.roi_manager import PeVleRoIManager
from pia_prod.AI.utils.init import logger
from pia_prod.AI.utils.log_templates import logging_send_alarm
from pia_prod.AI.utils.utils import get_event_type


class PeVle2StageAsyncService(ServiceBase):
    """PE 1단계 + Validation Server fire-and-forget 위임 (pe_vqa_2stage 미러)."""

    def __init__(self, analysis_data_queue: Queue):
        # Asyncio loop on a daemon thread (mirrors pe_vqa_2stage).
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="pe_vle_async_loop",
            daemon=True,
        )
        self._loop_thread.start()

        # httpx AsyncClient must be created on the loop.
        future = asyncio.run_coroutine_threadsafe(self._init_http_client(), self._loop)
        future.result(timeout=10)

        super().__init__(analysis_data_queue)
        self.alarm_event_manager.prepare_vectors_on_gpu(self.category_txt_vectors)
        self.is_needed_cvt_color = True

    async def _init_http_client(self):
        # Local re-import keeps tests' importlib.reload(config) effective.
        import pia_prod.AI.modules.pe_vle_2stage_async.config as _cfg
        self._http_client = httpx.AsyncClient(
            base_url=_cfg.VALIDATION_SERVER_URL,
            timeout=_cfg.VALIDATION_SERVER_TIMEOUT,
        )

    def __del__(self):
        if hasattr(self, "_http_client"):
            future = asyncio.run_coroutine_threadsafe(
                self._http_client.aclose(), self._loop
            )
            try:
                future.result(timeout=5)
            except Exception:
                pass
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ──────────────────────────────────────────────────────────────────
    # Stage 1 setup
    # ──────────────────────────────────────────────────────────────────
    def _init_values(self):
        self.category_txt_vectors = defaultdict()
        self.stream_vector_queues = defaultdict(lambda: deque(maxlen=TEMPORAL_SIZE))

    def _load_model(self):
        from pia_prod.AI.modules.perception_encoder.trt_load import TRTInference

        self.model = TRTInference(PERCEPTION_ENCODER_TRT_PATH)
        self.image_size = IMG_SIZE
        self._init_default_values()

    def _init_default_values(self):
        self._get_txt_vector_group_by_category()
        zero_mask_vec = self.model(
            torch.zeros(
                size=(1, 3, self.image_size[0], self.image_size[1]),
                dtype=torch.float32,
                device=DEVICE,
            )
        )
        self.zero_mask_vec = zero_mask_vec / zero_mask_vec.norm(dim=-1)

    def _load_roi_manager(self):
        return PeVleRoIManager()

    def _get_txt_vector_group_by_category(self):
        from pia_prod.AI.modules.perception_encoder.prompts import load_text_feature

        ID_list, class_list, prompt_list, text_features = load_text_feature(
            PERCEPTION_ENCODER_TXT_FEATURE_PATH, DEVICE
        )
        self.category_txt_vectors["ids"] = ID_list
        self.category_txt_vectors["vectors"] = text_features
        self.category_txt_vectors["class_list"] = class_list
        self.category_txt_vectors["prompt_list"] = prompt_list

    def _load_event_manager(self):
        return PeVleEventManager()

    # ──────────────────────────────────────────────────────────────────
    # Stage 1 detect — byte-parallel to pe_vqa_2stage._detect.
    # ──────────────────────────────────────────────────────────────────
    def _detect(self, **datas) -> Optional[dict]:
        batches = datas[BATCHES_KEY]
        stream_ids = datas[STREAM_IDS_KEY]
        user_params = datas[USER_PARAMS_KEY]

        if not self.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)

        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        image_cuda = preprocess_image(cropped_batches)
        visual_vectors = self.model(image_cuda)
        for stream_id, visual_vector in zip(stream_ids, visual_vectors):
            while self.stream_vector_queues[stream_id].__len__() < TEMPORAL_SIZE:
                self.stream_vector_queues[stream_id].append(self.zero_mask_vec)
            self.stream_vector_queues[stream_id].append(visual_vector)

        alarms, _ = self.alarm_event_manager(
            self.stream_vector_queues, stream_ids, user_params
        )

        if len(alarms) > 0:
            return {
                ALARMS_KEY: alarms,
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
        return None

    # ──────────────────────────────────────────────────────────────────
    # send_alarm — pe_vqa_2stage style override.
    #
    # PE_VLE_VALIDATION_ENABLED=False → match_outputs() for every alarm.
    # PE_VLE_VALIDATION_ENABLED=True  → category in TWO_STEP_CATEGORIES
    #     delegates to the validation server fire-and-forget. Other
    #     categories publish directly.
    # ──────────────────────────────────────────────────────────────────
    def send_alarm(self, results: dict):
        alarms = results[ALARMS_KEY]
        batches = results[BATCHES_KEY]
        stream_ids = results[STREAM_IDS_KEY]
        user_params = results[USER_PARAMS_KEY]
        is_needed_cvt_color = results[IS_NEEDED_CVT_COLOR_KEY]

        if not len(alarms):
            return

        alarms = self.get_alarm_with_uuid(alarms)

        for idx, (stream_id, (is_start, event_uuid)) in enumerate(alarms.items()):
            if "__" in stream_id:
                stream_id_clean, category_id = stream_id.split("__", 1)
            else:
                category_id = None
                for event_key in [CV_EVENT_KEY, RET_EVENT_KEY, VQA_EVENT_KEY]:
                    if event_key in user_params[idx][USER_PARAM_KEY]:
                        keys = user_params[idx][USER_PARAM_KEY][event_key].keys()
                        if len(keys):
                            category_id = list(keys)[0]
                            break
                stream_id_clean = stream_id

            batch_idx = [i for i, v in enumerate(stream_ids) if v == stream_id_clean][0]
            user_param = user_params[batch_idx]
            logging_send_alarm(stream_id_clean, is_start, category_id, event_uuid=event_uuid)

            # Thumbnail prep — same shape as pe_vqa_2stage. After this block
            # `thumbnail` is BGR (or None for is_start=False), ready for
            # match_outputs OR for cv2.imencode in the delegated payload.
            thumbnail = batches[batch_idx] if is_start else None
            if thumbnail is not None and isinstance(thumbnail, torch.Tensor):
                thumbnail = thumbnail.detach().contiguous().cpu().numpy()
            if is_needed_cvt_color and thumbnail is not None:
                thumbnail = thumbnail[..., ::-1]

            should_delegate = (
                PE_VLE_VALIDATION_ENABLED and category_id in TWO_STEP_CATEGORIES
            )
            if should_delegate:
                event_type = get_event_type(user_param)
                payload = self._build_validation_payload(
                    thumbnail=thumbnail,
                    is_start=is_start,
                    category_name=category_id,
                    stream_id=stream_id_clean,
                    event_uuid=event_uuid,
                    user_param=user_param,
                    event_type=event_type,
                )
                asyncio.run_coroutine_threadsafe(
                    self._send_to_validation_server(payload),
                    self._loop,
                )
            else:
                match_outputs(
                    thumbnail,
                    is_start,
                    user_param,
                    self.frame_cnt,
                    event_uuid=event_uuid,
                    category_name=category_id,
                )

    # ──────────────────────────────────────────────────────────────────
    # Validation payload — identical schema to pe_vqa_2stage.
    # ──────────────────────────────────────────────────────────────────
    def _build_validation_payload(
        self, thumbnail, is_start, category_name, stream_id,
        event_uuid, user_param, event_type,
    ) -> dict:
        thumbnail_b64 = None
        if thumbnail is not None:
            _, buf = cv2.imencode(".jpg", thumbnail)
            thumbnail_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        return {
            "thumbnail_b64": thumbnail_b64,
            "is_start": is_start,
            "category_name": category_name,
            "stream_id": stream_id,
            "event_uuid": event_uuid,
            "event_type": event_type,
            "user_param": user_param,
        }

    async def _send_to_validation_server(self, payload: dict):
        try:
            resp = await self._http_client.post(
                VALIDATION_SERVER_ENDPOINT,
                json=payload,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"[PeVle] validation server 전송 실패: {e}")
