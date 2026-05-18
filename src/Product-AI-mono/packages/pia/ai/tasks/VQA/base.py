from abc import abstractmethod
import os
from typing import List, Union

import numpy as np
import torch
from pia.ai.base import PiaConfigBase, PiaModelBase


class VQAConfig(PiaConfigBase):
    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3-2B",
        device: str = "cuda",
        n_gpus: int = torch.cuda.device_count(),
        input_size: int = 448,
        frame_per_tile_max_num: int = 12,
        max_new_tokens: int = 13,
        data_type: str = "bfloat16",
        save_cache_dir: str = "assets",
        api_host : str ="127.0.0.1",
        api_port : int = 9997,
        num_segments : int = 12,
        trt_build: dict = None,
    ):
        """
        VQA 모델 공통 설정.

        trt_build:
            TRT-LLM 전용 빌드 옵션을 담는 dict.
            - 기본값은 테스트 코드 경로/옵션을 기준으로 설정됨.
            - 외부에서 일부 키만 넘기면 기본값 위에 덮어쓰기됨.
        """
        super().__init__(model_path, device)
        self.input_size = input_size
        self.n_gpus = n_gpus
        self.frame_per_tile_max_num = frame_per_tile_max_num
        self.max_new_tokens = max_new_tokens
        self.data_type = self._check_data_type(data_type)
        self.save_cache_dir = save_cache_dir
        self.api_host = api_host
        self.api_port = api_port
        self.api_url = f"http://{api_host}:{api_port}/v1/chat/completions"
        self.num_segments = num_segments
        self.trt_build = self._default_trt_build()
        if trt_build:
            self.trt_build.update(trt_build)

    @staticmethod
    def _check_data_type(data_type):
        if data_type == "float16":
            return torch.float16
        elif data_type == "float32":
            return torch.float32
        elif data_type == "bfloat16":
            return torch.bfloat16
        else:
            raise ValueError("data_type must be float16, float32 or bfloat16")

    @staticmethod
    def _check_device_type(device):
        if "cuda" in device or "gpu" in device:
            return "cuda"
        raise ValueError("Device must be cuda")

    @staticmethod
    def _default_trt_build() -> dict:
        """
        TRT-LLM 빌드 기본값(테스트 코드 기준).

        경로는 repo 기준 상대 경로로 지정되며,
        실제 사용 시 절대 경로로 해석된다.

        Returns:
            TRT-LLM 빌드 설정 딕셔너리. 주요 키:
            - hf_repo_id: HF 모델 repo id
            - assets_dir/assets_model_dir/assets_image_dir: assets 경로
            - model_dir: 모델 저장 경로
            - vit_onnx_file/vit_trt_file/vit_image_path: VIT 관련 경로
            - vit_dtype/min/opt/max bs: VIT 엔진 설정
            - tllm_checkpoint_dir: TRT-LLM 체크포인트 경로
            - llm_dtype/llm_*: LLM 엔진 설정
            - trt_engine_dir: TRT 엔진 경로
        """
        hf_repo_id = "OpenGVLab/InternVL3-2B"
        assets_dir = "assets"
        assets_model_dir = os.path.join(assets_dir, "models")
        assets_image_dir = os.path.join(assets_dir, "images")
        model_dir = os.path.join(assets_model_dir, hf_repo_id)
        return {
            # ---- assets / paths ----
            "hf_repo_id": hf_repo_id,  # HF repo id(다운로드 기준)
            "assets_dir": assets_dir,  # assets 기본 경로
            "assets_model_dir": assets_model_dir,  # assets 하위 모델 경로
            "assets_image_dir": assets_image_dir,  # VIT 빌드용 샘플 이미지 경로
            "model_dir": model_dir,  # assets 기준 HF 모델 경로
            "vit_onnx_file": os.path.join(model_dir, "vision_encoder_bfp16.onnx"),  # VIT ONNX 파일
            "vit_trt_file": os.path.join(model_dir, "vision_encoder_bfp16.trt"),  # VIT TRT 엔진 파일
            "vit_image_path": os.path.join(assets_image_dir, "frame_0.jpg"),  # VIT 빌드용 샘플 이미지
            # ---- VIT engine ----
            "vit_dtype": "bfloat16",  # VIT 정밀도
            "vit_min_bs": 1,  # VIT 최소 배치(타일 수 합). sum(타일 수) >= 1
            "vit_opt_bs": 13,  # VIT 최적 배치(타일 수 합). 예: 1080p 2장 ≈ 9*2=18
            "vit_max_bs": 32,  # VIT 최대 배치(타일 수 합). sum(타일 수) <= max_bs
            # ---- LLM checkpoint/engine ----
            "tllm_checkpoint_dir": os.path.join(model_dir, "tllm_checkpoint"),  # TRT-LLM 체크포인트 경로
            "llm_dtype": "bfloat16",  # LLM 정밀도
            "load_model_on_cpu": True,  # 변환 시 HF 가중치 CPU 로드
            "disable_rope_scaling": True,  # 변환 시 RoPE scaling 비활성화
            # ---- LLM build limits ----
            "llm_max_batch_size": 2,  # LLM 엔진 최대 배치(동시 처리 샘플 수)
            "llm_max_input_len": 7312,  # 입력 최대 토큰(텍스트 + 이미지 토큰 + <img>/<\\img>)
            "llm_max_seq_len": 7326,  # 입력 + max_new_tokens(= llm_max_input_len + max_new_tokens)
            "llm_max_num_tokens": 20000,  # 배치 전체 토큰 예산(보통 llm_max_batch_size*llm_max_seq_len 이상)
            "llm_gemm_plugin": "bfloat16",  # GEMM 플러그인 정밀도
            "llm_max_multimodal_len": 23040,  # 멀티모달(이미지) 토큰 합 상한. sum(타일 수)*256
            # ---- runtime behavior ----
            "llm_force_single_batch": False,  # 배치 1 강제
            "llm_batch_use_executor": True,  # 배칭에 executor 사용
            "llm_batch_use_async": True,  # executor 비동기 경로 허용
            "llm_use_input_token_extra_ids": False,  # extra_ids 기반 KV cache reuse
            "trt_engine_dir": os.path.join(model_dir, "trt_engines"),  # TRT 엔진 출력 경로
        }


class VQAModelBase(PiaModelBase):
    def __init__(self, config: VQAConfig):
        self.config = config

    @abstractmethod
    def _load_model(self, model_path: str):
        pass

    @abstractmethod
    def preprocess(self, image: Union[str, np.ndarray], prompts: Union[str, List[str]]):
        pass

    @abstractmethod
    def forward(
        self, image: Union[str, np.ndarray], prompts: Union[str, List[str]]
    ) -> List[str]:
        pass
