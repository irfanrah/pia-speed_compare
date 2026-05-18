import asyncio
import base64
import threading
from queue import Queue
from typing import List

import cv2
import httpx

from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    USER_PARAM_KEY,
    VQA_EVENT_KEY,
)
from pia_prod.AI.modules.soil_qwen35.config import (
    VLLM_API_URL,
    VLLM_MODEL,
    VLLM_MAX_TOKENS,
    VLLM_TEMPERATURE,
    VLLM_TIMEOUT,
    ALL_CATEGORIES,
)
from pia_prod.AI.modules.soil_qwen35.event import SoilQwen35EventManager
from pia_prod.AI.modules.soil_qwen35.prompts import get_prompt
from pia_prod.AI.modules.soil_qwen35.roi_manager import SoilQwen35RoIManager


class SoilQwen35Service(ServiceBase):
    """
    Soil Qwen3.5 VQA 서비스.

    외부 vLLM 서버(Qwen3.5-0.8B)에 이미지+프롬프트를 보내
    yes/no 응답으로 알람을 판정하는 독립형 VQA 모듈.
    """

    def __init__(self, analysis_data_queue: Queue):
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="soil_qwen35_async_loop",
            daemon=True,
        )
        self._loop_thread.start()

        future = asyncio.run_coroutine_threadsafe(self._init_http_client(), self._loop)
        future.result(timeout=10)

        super().__init__(analysis_data_queue)

    async def _init_http_client(self):
        # 테스트에서 importlib.reload(config)로 동적 재로딩을 지원하기 위해 지역 import 사용
        import pia_prod.AI.modules.soil_qwen35.config as _cfg
        self._http_client = httpx.AsyncClient(timeout=_cfg.VLLM_TIMEOUT)
        self._api_url = f"{_cfg.VLLM_API_URL.rstrip('/')}/chat/completions"

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

    def _init_values(self):
        pass

    def _load_model(self):
        pass

    def _load_event_manager(self):
        return SoilQwen35EventManager()

    def _load_roi_manager(self):
        return SoilQwen35RoIManager()

    def _match_category2prompt(self, batches, user_params):
        re_batches = []
        prompts = []
        categories = []
        for frame, user_param in zip(batches, user_params):
            events = user_param[USER_PARAM_KEY][VQA_EVENT_KEY]
            temp_categories = []
            for category in events:
                if category not in ALL_CATEGORIES:
                    continue
                prompt = get_prompt(category)
                if prompt is None:
                    continue
                temp_categories.append(category)
                re_batches.append(frame)
                prompts.append(prompt)
            categories.append(temp_categories)
        return re_batches, prompts, categories

    async def _call_vllm_async(self, image, prompt: str) -> str:
        _, buf = cv2.imencode('.jpg', image)
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        payload = {
            "model": VLLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": VLLM_MAX_TOKENS,
            "temperature": VLLM_TEMPERATURE,
        }
        resp = await self._http_client.post(self._api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _batch_inference(self, batches: List, prompts: List[str]) -> List[str]:
        tasks = [self._call_vllm_async(img, prompt) for img, prompt in zip(batches, prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        responses = []
        for r in results:
            if isinstance(r, Exception):
                from pia_prod.AI.utils.init import logger
                logger.error(f"[SoilQwen35] vLLM API 호출 실패: {r}")
                responses.append("no")
            else:
                responses.append(r)
        return responses

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        batches = self.roi_manager.process_batches_with_roi(batches, user_params)

        re_batches, prompts, categories = self._match_category2prompt(batches, user_params)

        if not prompts:
            return None

        future = asyncio.run_coroutine_threadsafe(
            self._batch_inference(re_batches, prompts), self._loop
        )
        responses = future.result(timeout=VLLM_TIMEOUT + 10)

        alarms = self.alarm_event_manager.update(responses, categories, stream_ids)

        if len(alarms) > 0:
            return {
                ALARMS_KEY: alarms,
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
        else:
            return None
