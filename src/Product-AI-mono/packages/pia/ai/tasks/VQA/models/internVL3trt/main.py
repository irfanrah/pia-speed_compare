import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as futures_thread
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import requests

from ...base import VQAConfig, VQAModelBase


class InternVL3trt(VQAModelBase):
    def __init__(self, config: VQAConfig):
        super().__init__(config)
        self.config = config
        self._load_model()

    def _load_model(self):
        self.model_name = "internVL3"
        self.model = "InternVL3trt"
        self.api_url = self.config.api_url
        self.tokenizer = None
        self.transform = None
        self.generation_config = None
        ### 임시 스레드 풀 테스트 변수
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._closed = False
        self.num_tokens = self.config.max_new_tokens

    def close(self):
        if not self._closed:
            # ✅ 안전 종료
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def _is_shutting_down(self) -> bool:
        # concurrent.futures 내부 플래그
        return bool(getattr(futures_thread, "_shutdown", False))

    def _safe_submit(self, fn, *args, **kwargs):

        if self._closed or self._is_shutting_down():
            # ✅ 종료 중: 새 스레드 제출 금지 → 동기 실행
            return fn(*args, **kwargs)

        try:
            fut = self._executor.submit(fn, *args, **kwargs)
            return fut  # caller가 fut.result() 할 수 있게 반환
        except RuntimeError as e:
            # ✅ 인터프리터 종료 중… 동기 경로로 재시도
            if "after interpreter shutdown" in str(e):
                return fn(*args, **kwargs)
            raise

    def preprocess(self, image: Union[str, np.ndarray], prompts: Union[str, List[str]]):
        pass  # 사용하지 않음

    @staticmethod
    def extract_response_text(response: Dict[str, Any]) -> str:
        """API 응.답 JSON에서 content 텍스트를 안전하게 추출합니다."""
        try:
            return response['result']['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            return f"Error parsing response: {response.get('error', 'Unknown error')}"

    @staticmethod
    def _encode_numpy_to_base64(image_array: np.ndarray) -> str:
        """BGR NumPy 배열을 RGB로 변환 후, JPEG 형식으로 인코딩하고 Base64 문자열로 변환합니다."""
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(f"이미지 모양이 (H,W,3) BGR 이어야 합니다. 현재: {image_array.shape}")
        # rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode('.jpg', image_array)

        if not success:
            raise ValueError("이미지를 JPEG 형식으로 인코딩하는 데 실패했습니다.")
        return base64.b64encode(buffer).decode('utf-8')

    def _video_send_request(self, video_array: np.ndarray, prompt: str, identifier: str, max_frames: int = 12) -> Dict[str, Any]:
        """
        비디오(T,H,W,C) NumPy 배열과 프롬프트에 대한 API 요청을 전송합니다.

        Args:
            video_array: (T,H,W,C) 형태의 비디오 배열. T는 프레임 수.
            prompt: 비디오에 대한 프롬프트
            identifier: 요청 식별자
            max_frames: API로 전송할 최대 프레임 수 (기본값: 8)

        Returns:
            API 응답 딕셔너리
        """
        start_time = time.perf_counter()

        if video_array.ndim != 4:
            return {"error": f"비디오는 4차원(T,H,W,C)이어야 합니다. 현재: {video_array.ndim}차원",
                    "path": identifier, "duration": f"{time.perf_counter() - start_time:.2f}s"}

        total_frames = video_array.shape[0]

        if total_frames <= max_frames:
            selected_indices = list(range(total_frames))
        else:
            selected_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

        base64_frames = []
        try:
            # TODO : [(T,H,W,C), (T,H,W,C), ...]  ← 프레임 배열 리스트로 구성하는게 나은가..?
            for idx in selected_indices:
                frame = video_array[idx]  # (H,W,C)
                # TODO : image 전처리
                if frame.shape[:2] != (448, 448):
                    frame = cv2.resize(frame, (448 , 448) , interpolation=cv2.INTER_CUBIC)
                base64_frame = self._encode_numpy_to_base64(frame)
                base64_frames.append(base64_frame)

        except Exception as e:
            duration = time.perf_counter() - start_time
            return {"error": f"프레임 인코딩 실패: {e}", "path": identifier, "duration": f"{duration:.2f}s"}

        frame_text = ""
        for i in range(len(base64_frames)):
            frame_text += f'Frame{i + 1}: <image>\n'

        final_prompt = frame_text + prompt

        content = [{"type": "text", "text": final_prompt}]
        for base64_frame in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
            })

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": self.num_tokens
        }

        try:
            # TODO : 아래 2라인 차이가 속도차이를 발생시킴 왜지? - 테스트 필요 -> 해보니깐 또 아님, 왜지?
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)
            # response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            duration = time.perf_counter() - start_time
            # print(f"비디오 요청 총 시간: {identifier} ({duration:.2f}s)")
            # print(f"요청 응답 시간 : {(et -st):.2f}")

            return {"result": response.json(), "path": identifier, "duration": f"{duration:.2f}s"}
        except requests.exceptions.RequestException as e:
            duration = time.perf_counter() - start_time
            # print(f"오류 발생 ({identifier}): {e}")
            return {"error": str(e), "path": identifier, "duration": f"{duration:.2f}s"}


    def _image_send_request(self, image_array: np.ndarray, prompt: str, identifier: str) -> Dict[str, Any]:
        """단일 이미지와 프롬프트에 대한 API 요청을 전송합니다."""
        start_time = time.perf_counter()
        prompt = "<image>\n" + prompt
        try:
            base64_image = self._encode_numpy_to_base64(image_array)
        except Exception as e:
            duration = time.perf_counter() - start_time
            return {"error": f"Image encoding failed: {e}", "path": identifier, "duration": f"{duration:.2f}s"}

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": self.num_tokens
        }

        try:
            # TODO : 아래 2라인 차이가 속도차이를 발생시킴 왜지? - 테스트 필요 -> 해보니깐 또 아님, 왜지? -> 추론시간 늘리는 원인 파악 필요
            # response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)

            response.raise_for_status()
            duration = time.perf_counter() - start_time
            # print(f"요청 완료: {identifier} ({duration:.2f}s)")
            return {"result": response.json(), "path": identifier, "duration": f"{duration:.2f}s"}
        except requests.exceptions.RequestException as e:
            duration = time.perf_counter() - start_time
            # print(f"오류 발생 ({identifier}): {e}")
            return {"error": str(e), "path": identifier, "duration": f"{duration:.2f}s"}

    def forward(self, images: Union[np.ndarray, List[np.ndarray]], prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        이미지/비디오와 프롬프트를 받아 모델의 응답을 반환합니다.

        입력 형태:
        - (H,W,C): 단일 이미지
        - (T,H,W,C): 단일 비디오
        - [(H,W,C), ...]: 이미지 리스트
        - [(T,H,W,C), ...]: 비디오 리스트
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                input_type = "single_image"
                media_list = [images]
            elif images.ndim == 4:
                input_type = "single_video"
                media_list = [images]
            else:
                raise ValueError(f"np.ndarray는 3차원 또는 4차원이어야 합니다. 현재: {images.ndim}차원")

        elif isinstance(images, list):
            if len(images) > 0:
                first_item = images[0]
                if first_item.ndim == 3:
                    input_type = "image_list"
                elif first_item.ndim == 4:
                    input_type = "video_list"
            media_list = images
        else:
            raise TypeError(f"images는 np.ndarray 또는 List[np.ndarray]여야 합니다. 현재: {type(images)}")

        batch_size = len(media_list)

        if isinstance(prompts, str):
            prompt_list = [prompts] * batch_size
        else:
            prompt_list = prompts

        if batch_size != len(prompt_list):
            raise ValueError(f"배치 개수({(batch_size)})와 프롬프트 개수({len(prompt_list)})가 일치하지 않습니다.")

        results = [None] * batch_size # 순서 보장을 위해 미리 공간 할당

        ###########################################################################
        # 병렬이 의미 없거나 종료 중이면 동기 루프
        use_threads = (batch_size > 1) and (not self._closed) and (not self._is_shutting_down())

        if not use_threads:
            for i, (media, prompt) in enumerate(zip(media_list, prompt_list)):
                if input_type in ["single_video", "video_list"]:
                    api_response = self._video_send_request(media, prompt, f"Video-{i}", getattr(self.config, "num_segments", 12))
                else:
                    api_response = self._image_send_request(media, prompt, f"Image-{i}")
                results[i] = self.extract_response_text(api_response)
            return results

        futs = []
        idxs = []
        for i, (media, prompt) in enumerate(zip(media_list, prompt_list)):
            identifier = f"{'Video' if input_type in ['single_video','video_list'] else 'Image'}-{i}"

            if input_type in ["single_video", "video_list"]:
                # video: 4개 인자 (media, prompt, identifier, max_frames)
                submitted = self._safe_submit(
                    self._video_send_request,
                    media,
                    prompt,
                    identifier,
                    getattr(self.config, "num_segments", 12)
                )
            else:
                # image: 3개 인자만 (media, prompt, identifier)
                submitted = self._safe_submit(
                    self._image_send_request,
                    media,
                    prompt,
                    identifier
                )

            if hasattr(submitted, "result"):  # 진짜 Future
                futs.append(submitted)
                idxs.append(i)
            else:
                # 종료 중 동기 실행 결과가 바로 리턴된 경우
                results[i] = self.extract_response_text(submitted)

        # 남은 Future 수거
        for fut, i in zip(futs, idxs):
            try:
                api_response = fut.result()
                results[i] = self.extract_response_text(api_response)
            except Exception as e:
                results[i] = f"Error: {e}"

        return results
