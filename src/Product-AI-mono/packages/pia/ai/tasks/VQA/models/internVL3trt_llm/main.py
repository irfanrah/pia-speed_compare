import datetime
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

from ...base import VQAConfig, VQAModelBase
from .config import (
    DEFAULT_SYSTEM_PROMPT,
    END_TOKEN_ID,
    IMAGE_TAG,
    IMG_BEGIN_TOKEN_ID,
    IMG_TOKEN,
    PAD_TOKEN_ID,
    RUNTIME_INFER_ARGS,
    SAMPLING_CONFIG,
    SKIP_SPECIAL_TOKENS,
)


class InternVL3trt_llm(VQAModelBase):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: VQAConfig):
        """InternVL3 TRT-LLM VQA 모델 인스턴스를 초기화한다.

        Args:
            config: VQAConfig 설정 객체. 모델 경로, 디바이스, 입력 크기,
                토큰/샘플링 설정 등을 포함한다.

        Raises:
            RuntimeError: 엔진/체크포인트 로딩에 실패한 경우.
            ValueError: config 값이 유효하지 않은 경우.
        """
        super().__init__(config)
        self.config = config
        self._load_model(config.model_path)

    def _load_model(self, model_path):
        """
        모델/엔진/토크나이저를 로드한다.

        동작 순서:
        1) 모델 경로 존재 여부 확인
        2) 없으면 HF 스냅샷 다운로드
        3) VIT TRT 엔진 없으면 빌드
        4) LLM 체크포인트 없으면 변환
        5) LLM TRT 엔진 없으면 빌드

        경로는 YAML이 아닌 실제 파일/폴더 기준으로 판단한다.

        Args:
            model_path: 사용자 설정에서 지정한 모델 경로. 상대/절대 경로 모두 허용.

        Raises:
            RuntimeError: 엔진/체크포인트/모델 경로 검증 실패 시.
            FileNotFoundError: 필요한 파일이 없을 때.
        """
        trt_build = self._normalize_trt_build()
        self.trt_build = trt_build

        assets_root = self._resolve_assets_root(trt_build)
        self.assets_root = assets_root

        model_dir = self._resolve_model_dir_from_config(model_path, trt_build, assets_root)
        model_dir = self._ensure_model_dir(model_dir, trt_build, assets_root)

        vit_onnx_file = self._resolve_path(trt_build.get("vit_onnx_file"), assets_root)
        vit_trt_file = self._resolve_path(trt_build.get("vit_trt_file"), assets_root)
        vit_image_path = self._resolve_path(trt_build.get("vit_image_path"), assets_root)
        tllm_checkpoint_dir = self._resolve_path(trt_build.get("tllm_checkpoint_dir"), assets_root)
        trt_engine_dir = self._resolve_path(trt_build.get("trt_engine_dir"), assets_root)

        self._ensure_vit_engine(
            model_dir=model_dir,
            onnx_file=vit_onnx_file,
            trt_file=vit_trt_file,
            image_path=vit_image_path,
            trt_build=trt_build,
        )
        self._ensure_llm_checkpoint(
            model_dir=model_dir,
            checkpoint_dir=tllm_checkpoint_dir,
            trt_build=trt_build,
        )
        self._ensure_llm_engine(
            checkpoint_dir=tllm_checkpoint_dir,
            output_dir=trt_engine_dir,
            trt_build=trt_build,
        )

        tokenizer_path = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )

        self.img_token = IMG_TOKEN
        self.img_begin_token_id = int(IMG_BEGIN_TOKEN_ID or 0)
        if not self.img_begin_token_id:
            self.img_begin_token_id = int(getattr(self.tokenizer, "vocab_size", 0))
        self.skip_special_tokens = set(SKIP_SPECIAL_TOKENS)
        self.end_token_id = END_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        if self.end_token_id is not None:
            self.end_token_id = int(self.end_token_id)
        if self.pad_token_id is not None:
            self.pad_token_id = int(self.pad_token_id)
        self.sampling_cfg = dict(SAMPLING_CONFIG)

        self.vit_runner = TrtVitRunner(vit_trt_file)
        self.engine_dir = trt_engine_dir
        
        self.use_cpp_runner = bool(getattr(self.config, "use_cpp_runner", True))
        self.llm_runner = None
        self.trt_executor = None
        self.llm_executor = None
        runtime_args = dict(RUNTIME_INFER_ARGS)
        self._runtime_args = runtime_args
        self.use_kv_cache_reuse = bool(
            self.trt_build.get("llm_use_input_token_extra_ids", False)
        )
        if self.use_cpp_runner:
            self.llm_runner = create_cpp_runner(self.engine_dir, runtime_args)
        else:
            self.trt_executor = get_trt_executor()
            self.llm_executor = create_executor(
                self.trt_executor,
                self.engine_dir,
                runtime_args,
                enable_kv_cache_reuse=self.use_kv_cache_reuse,
            )

        self.transform = build_transform(self.config.input_size, self.IMAGENET_MEAN, self.IMAGENET_STD)
        self.max_new_tokens = self.config.max_new_tokens

    def preprocess(self, image: Union[str, np.ndarray], prompts: Union[str, List[str]]):
        """전처리 훅. 현재 구현에서는 사용하지 않는다.

        Args:
            image: 이미지 경로 또는 HWC numpy 배열.
            prompts: 단일 프롬프트 또는 프롬프트 리스트.
        """
        pass  # not used

    def forward(
        self, images: Union[np.ndarray, List[np.ndarray]], prompts: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """이미지와 프롬프트를 받아 VQA 추론을 수행한다.

        Args:
            images: 단일 HWC numpy 배열 또는 HWC numpy 배열 리스트.
            prompts: 단일 프롬프트 또는 프롬프트 리스트.
                단일 문자열인 경우 이미지 수만큼 복제된다.

        Returns:
            단일 문자열 또는 문자열 리스트. 이미지 개수에 맞춘 응답을 반환한다.

        Raises:
            ValueError: 이미지/프롬프트 개수 불일치, 형식 오류 시.
            TypeError: 이미지 타입이 numpy 배열이 아닐 때.
            RuntimeError: 엔진 추론 실패, 전처리 실패 등 내부 오류.
        """
        if isinstance(images, np.ndarray):
            if images.ndim != 3:
                raise ValueError(f"images must be HWC np.ndarray; got ndim={images.ndim}")
            image_list = [images]
            single = True
        elif isinstance(images, list):
            if not images:
                raise ValueError("images list is empty")
            if not all(isinstance(img, np.ndarray) and img.ndim == 3 for img in images):
                raise ValueError("all images must be HWC numpy arrays")
            image_list = images
            single = False
        else:
            raise TypeError(f"images must be np.ndarray or List[np.ndarray], got {type(images)}")

        if isinstance(prompts, str):
            prompt_list = [prompts] * len(image_list)
        else:
            prompt_list = prompts
        if len(prompt_list) != len(image_list):
            raise ValueError(
                f"batch mismatch: images={len(image_list)} prompts={len(prompt_list)}"
            )

        all_patches, patch_counts = self._image_preprocess_batch(image_list)
        self._validate_tile_counts(patch_counts)

        need_extra_ids = self.use_kv_cache_reuse or (not self.use_cpp_runner)
        if single:
            img_hash = hash_image_bytes(image_list[0]) if need_extra_ids else None
            return self._infer_single(
                all_patches[0], patch_counts[0], prompt_list[0], img_hash
            )

        img_hashes = None
        if need_extra_ids:
            img_hashes = [hash_image_bytes(img) for img in image_list]
        return self._infer_batch(all_patches, patch_counts, prompt_list, img_hashes)

    def _image_preprocess_batch(
        self, images: List[np.ndarray],
    ) -> tuple[List[torch.Tensor], List[int]]:
        """
        배치 이미지를 전처리하여 이미지별 타일 텐서 리스트와 타일 개수 목록을 반환한다.

        Args:
            images: HWC 형식의 numpy 배열 리스트.

        Returns:
            (all_patches, patch_counts)
            - all_patches: 이미지별 타일 텐서 리스트. 각 원소는 (N, C, H, W).
            - patch_counts: 각 이미지의 타일 개수 리스트.

        Raises:
            RuntimeError: 타일이 생성되지 않은 경우.
        """
        all_patches = []
        patch_counts = []

        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            tiles = dynamic_preprocess(
                image,
                image_size=self.config.input_size,
                use_thumbnail=True,
                max_num=int(self.config.frame_per_tile_max_num),
            )
            if not tiles:
                raise RuntimeError("이미지 타일이 생성되지 않았습니다.")
            patch_counts.append(len(tiles))
            tile_tensors = torch.stack([self.transform(tile) for tile in tiles])
            all_patches.append(tile_tensors)

        return all_patches, patch_counts

    def _infer_single(
        self,
        pixel_values: torch.Tensor,
        patch_count: int,
        prompt: str,
        img_hash: Optional[int] = None,
    ) -> str:
        """
        단일 이미지 추론 경로. C++ runner 기준으로 가장 안정적인 경로다.

        Args:
            pixel_values: 이미지 타일 텐서 (N, C, H, W).
            patch_count: 타일 개수.
            prompt: 사용자 프롬프트.

        Returns:
            모델 응답 문자열.

        Raises:
            RuntimeError: VIT 출력 형태가 예상과 다르거나 추론 실패 시.
        """
        base_prompt = build_prompt(prompt)
        prompt_with_img = replace_image_tokens(base_prompt, [patch_count], self.img_token)
        input_ids = encode_with_img_tokens(
            self.tokenizer, prompt_with_img, self.img_token, self.img_begin_token_id
        )

        images_tensor = pixel_values.to(
            device="cuda",
            dtype=trt_dtype_to_torch(self.vit_runner.inp_dtype),
        )
        vit_out = self.vit_runner.infer(images_tensor)
        if vit_out.ndim == 3:
            vit_out = vit_out.reshape(vit_out.shape[0] * vit_out.shape[1], vit_out.shape[2])
        elif vit_out.ndim != 2:
            raise RuntimeError(f"Unexpected VIT output shape: {vit_out.shape}")

        extra_ids = None
        if img_hash is not None:
            extra_ids = build_extra_ids(input_ids, self.img_begin_token_id, img_hash)

        if self.use_cpp_runner:
            prompt_table = vit_out.unsqueeze(0) if vit_out.ndim == 2 else vit_out
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.int32)
            output_ids = self.llm_runner.generate(
                [input_ids_tensor],
                max_new_tokens=self.max_new_tokens,
                end_id=self.end_token_id,
                pad_id=self.pad_token_id,
                prompt_table=prompt_table,
                input_token_extra_ids=extra_ids,
                top_k=self.sampling_cfg.get("top_k", 1),
                top_p=self.sampling_cfg.get("top_p", 0.1),
                temperature=self.sampling_cfg.get("temperature", 0.1),
                return_dict=False,
            )
            output_ids = normalize_output_ids(output_ids)
        else:
            # NOTE: 거의 사용하지 않는 폴백 경로. C++ runner가 없는 환경을 대비해 유지한다.
            ptuning_tensor = make_trt_tensor(self.trt_executor, vit_out)
            ptuning_config = make_prompt_tuning_config(
                self.trt_executor, ptuning_tensor, extra_ids
            )
            sampling_config = make_sampling_config(self.trt_executor, self.sampling_cfg)
            output_config = make_output_config(self.trt_executor, exclude_input_from_output=True)
            request = create_request(
                self.trt_executor,
                input_ids,
                self.max_new_tokens,
                sampling_config,
                output_config,
                ptuning_config,
                self.end_token_id,
                self.pad_token_id,
            )
            request_id = enqueue_request(self.llm_executor, request)
            response = wait_for_response(self.llm_executor, request_id, timeout_ms=60000)
            output_ids = extract_output_token_ids(response)
            output_ids = normalize_output_ids(output_ids)

        output_ids = strip_input_prefix(output_ids, input_ids)
        output_ids = trim_trailing_eos_pad(output_ids, self.end_token_id, self.pad_token_id)
        text = decode_tokens(
            self.tokenizer, output_ids, self.skip_special_tokens, self.img_begin_token_id
        )
        return text.strip()

    def _validate_tile_counts(self, patch_counts: List[int]) -> None:
        """
        전처리 후 실제 패치(타일) 수가 엔진 제약을 초과하는지 검증한다.
        초과 시 RuntimeError를 발생시킨다 (타일 수를 줄이면 성능이 하락하므로 에러 처리).

        Args:
            patch_counts: 이미지별 타일 개수 리스트.

        Raises:
            RuntimeError: 타일 수가 엔진 제약을 초과하는 경우.
        """
        max_tiles = int(self.config.frame_per_tile_max_num)
        for idx, count in enumerate(patch_counts):
            if count > max_tiles:
                raise RuntimeError(
                    f"이미지 {idx}의 타일 수({count})가 최대 허용값({max_tiles})을 초과합니다. "
                    f"이미지 해상도를 줄이거나 frame_per_tile_max_num을 늘려주세요."
                )

        total_vision_tokens = sum(patch_counts) * 256
        max_prompt_tokens = int(self.trt_build.get("llm_max_multimodal_len", 0) or 0)
        if hasattr(self, "llm_runner") and hasattr(self.llm_runner, "max_prompt_embedding_table_size"):
            engine_prompt_max = int(self.llm_runner.max_prompt_embedding_table_size or 0)
            if engine_prompt_max > 0:
                max_prompt_tokens = engine_prompt_max
        if max_prompt_tokens > 0 and total_vision_tokens > max_prompt_tokens:
            raise RuntimeError(
                f"배치 전체 비전 토큰 수({total_vision_tokens})가 "
                f"max_prompt_embedding_table_size({max_prompt_tokens})를 초과합니다. "
                f"배치 크기를 줄이거나 이미지 해상도를 낮춰주세요."
            )

        vit_max_bs = int(self.trt_build.get("vit_max_bs", 0) or 0)
        total_patches = sum(patch_counts)
        if vit_max_bs > 0 and total_patches > vit_max_bs:
            raise RuntimeError(
                f"배치 전체 패치 수({total_patches})가 "
                f"VIT 엔진 max_batch_size({vit_max_bs})를 초과합니다."
            )

    def _infer_batch(
        self,
        all_patches: List[torch.Tensor],
        patch_counts: List[int],
        prompts: List[str],
        img_hashes: Optional[List[int]] = None,
    ) -> List[str]:
        """
        배치 단위로 VIT 추론 + TRT-LLM 생성을 수행한다.
        multimodal_batch_test.py의 검증된 배치 로직을 이식한 구현.

        Args:
            all_patches: 이미지별 타일 텐서 리스트.
            patch_counts: 이미지별 타일 개수 리스트.
            prompts: 이미지별 프롬프트 리스트.

        Returns:
            배치 크기만큼의 응답 문자열 리스트.

        Raises:
            RuntimeError: VIT 출력 형태가 예상과 다르거나 추론 실패 시.
        """
        batch_size = len(prompts)
        vit_dtype = trt_dtype_to_torch(self.vit_runner.inp_dtype)

        # --- Step 1: VIT 배치 추론 (모든 이미지 패치를 concat하여 한 번에 처리) ---
        combined = torch.cat(all_patches, dim=0).to(device="cuda", dtype=vit_dtype)
        vit_out = self.vit_runner.infer(combined)
        torch.cuda.synchronize()

        hidden_size = vit_out.shape[-1]
        per_image_vit = []
        if vit_out.ndim == 3:
            offset = 0
            for count in patch_counts:
                chunk = vit_out[offset:offset + count]
                chunk = chunk.reshape(count * 256, hidden_size)
                per_image_vit.append(chunk)
                offset += count
        elif vit_out.ndim == 2:
            offset = 0
            for count in patch_counts:
                length = count * 256
                per_image_vit.append(vit_out[offset:offset + length])
                offset += length
        else:
            raise RuntimeError(f"Unexpected VIT output shape: {vit_out.shape}")

        # --- Step 2: 프롬프트 빌드 ---
        all_input_ids = []
        for prompt_text, count in zip(prompts, patch_counts):
            base = build_prompt(prompt_text)
            expanded = replace_image_tokens(base, [count], self.img_token)
            ids = encode_with_img_tokens(
                self.tokenizer, expanded, self.img_token, self.img_begin_token_id
            )
            all_input_ids.append(ids)

        all_extra_ids = None
        if img_hashes:
            all_extra_ids = [
                build_extra_ids(ids, self.img_begin_token_id, img_hash)
                for ids, img_hash in zip(all_input_ids, img_hashes)
            ]

        # --- Step 3: LLM 배치 추론 ---
        if self.use_cpp_runner:
            # padded prompt table 생성
            max_len = max(v.shape[0] for v in per_image_vit)
            padded = torch.zeros(
                batch_size, max_len, hidden_size,
                dtype=per_image_vit[0].dtype, device=per_image_vit[0].device,
            )
            for i, v in enumerate(per_image_vit):
                padded[i, :v.shape[0], :] = v
            prompt_tasks = ",".join(str(i) for i in range(batch_size))

            batch_input_tensors = [
                torch.tensor(ids, dtype=torch.int32) for ids in all_input_ids
            ]
            output_ids_batch = self.llm_runner.generate(
                batch_input_tensors,
                max_new_tokens=self.max_new_tokens,
                end_id=self.end_token_id,
                pad_id=self.pad_token_id,
                prompt_table=padded,
                prompt_tasks=prompt_tasks,
                input_token_extra_ids=all_extra_ids if all_extra_ids else None,
                top_k=self.sampling_cfg.get("top_k", 1),
                top_p=self.sampling_cfg.get("top_p", 0.1),
                temperature=self.sampling_cfg.get("temperature", 0.1),
                return_dict=False,
            )
            output_ids_list = normalize_output_ids_batch(output_ids_batch)
        else:
            self._ensure_executor()
            sampling_config = make_sampling_config(self.trt_executor, self.sampling_cfg)
            output_config = make_output_config(
                self.trt_executor, exclude_input_from_output=True
            )
            request_ids = []
            for idx in range(batch_size):
                extra = all_extra_ids[idx] if all_extra_ids else None
                ptuning_tensor = make_trt_tensor(self.trt_executor, per_image_vit[idx])
                ptuning_config = make_prompt_tuning_config(
                    self.trt_executor, ptuning_tensor, extra
                )
                request = create_request(
                    self.trt_executor,
                    all_input_ids[idx],
                    self.max_new_tokens,
                    sampling_config,
                    output_config,
                    ptuning_config,
                    self.end_token_id,
                    self.pad_token_id,
                )
                rid = enqueue_request(self.llm_executor, request)
                request_ids.append(normalize_request_id(rid))

            responses = wait_for_all_responses(
                self.llm_executor, request_ids, timeout_ms=60000
            )
            output_ids_list = [
                normalize_output_ids(extract_output_token_ids(r)) for r in responses
            ]

        output_ids_list = [
            strip_input_prefix(output_ids, input_ids)
            for output_ids, input_ids in zip(output_ids_list, all_input_ids)
        ]

        # --- Step 4: 디코딩 ---
        outputs = []
        for output_ids in output_ids_list:
            output_ids = trim_trailing_eos_pad(
                output_ids, self.end_token_id, self.pad_token_id
            )
            text = decode_tokens(
                self.tokenizer, output_ids, self.skip_special_tokens, self.img_begin_token_id
            )
            outputs.append(text.strip())
        return outputs

    def _ensure_executor(self) -> None:
        """
        C++ runner를 쓰는 환경에서도 배치 executor 경로가 필요할 때를 대비해
        executor를 지연 생성한다.

        Raises:
            RuntimeError: executor 생성 실패 시.
        """
        if self.trt_executor is not None and self.llm_executor is not None:
            return
        self.trt_executor = get_trt_executor()
        self.llm_executor = create_executor(
            self.trt_executor,
            self.engine_dir,
            self._runtime_args,
            enable_kv_cache_reuse=self.use_kv_cache_reuse,
        )

    @staticmethod
    def _resolve_assets_root(trt_build: dict) -> Path:
        """assets 루트를 계산한다.

        Args:
            trt_build: TRT 빌드 설정 딕셔너리.

        Returns:
            assets 루트 경로(Path).
        """
        assets_dir = trt_build.get("assets_dir", "assets")
        if assets_dir:
            candidate = Path(assets_dir)
            if not candidate.is_absolute():
                candidate = Path(os.getcwd()) / candidate
            if candidate.is_dir():
                return candidate
        cwd_assets = Path(os.getcwd()) / "assets"
        return cwd_assets

    @staticmethod
    def _resolve_path(path: Optional[str], assets_root: Path) -> Optional[str]:
        """상대경로/`/assets` 경로를 실제 경로로 변환한다.

        Args:
            path: 원본 경로 문자열. None이면 그대로 반환.
            assets_root: assets 기준 경로.

        Returns:
            절대 경로 문자열 또는 None.
        """
        if not path:
            return path
        if str(path).startswith("/assets/"):
            rel = str(path)[len("/assets/") :]
            path = str(assets_root / rel)
        p = Path(path)
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        return str(p)

    def _normalize_trt_build(self) -> dict:
        """trt_build 기본값 + 사용자 override 병합.

        Returns:
            정규화된 TRT 빌드 설정 딕셔너리.
        """
        base = {}
        if hasattr(self.config, "trt_build") and isinstance(self.config.trt_build, dict):
            base.update(self.config.trt_build)
        return base

    def _resolve_model_dir_from_config(self, model_path: str, trt_build: dict, assets_root: Path) -> str:
        """모델 경로를 결정한다(YAML 경로는 무시).

        Args:
            model_path: 사용자가 넘긴 model_path.
            trt_build: TRT 빌드 설정.
            assets_root: assets 루트 경로.

        Returns:
            모델 디렉터리 경로 문자열.
        """
        if model_path and not model_path.endswith((".yml", ".yaml")):
            candidate = self._resolve_path(model_path, assets_root)
            if candidate and Path(candidate).is_dir():
                return candidate
        model_dir = trt_build.get("model_dir")
        if model_dir:
            return self._resolve_path(model_dir, assets_root)
        assets_model_dir = trt_build.get("assets_model_dir")
        if assets_model_dir:
            assets_model_dir = self._resolve_path(assets_model_dir, assets_root)
            repo = trt_build.get("hf_repo_id", "OpenGVLab/InternVL3-2B")
            return str(Path(assets_model_dir) / repo)
        repo = trt_build.get("hf_repo_id", "OpenGVLab/InternVL3-2B")
        return str(assets_root / "models" / repo)

    def _ensure_model_dir(self, model_dir: str, trt_build: dict, assets_root: Path) -> str:
        """
        모델 디렉토리가 없으면 HF 스냅샷을 다운로드한다.

        Args:
            model_dir: 기대 모델 디렉터리 경로.
            trt_build: TRT 빌드 설정.
            assets_root: assets 루트 경로.

        Returns:
            실제 존재하는 모델 디렉터리 경로.

        Raises:
            RuntimeError: 모델 다운로드/검증 실패 시.
        """
        if model_dir and Path(model_dir).is_dir():
            return model_dir
        hf_repo_id = trt_build.get("hf_repo_id", "OpenGVLab/InternVL3-2B")
        assets_model_dir = trt_build.get("assets_model_dir")
        if assets_model_dir:
            assets_model_dir = self._resolve_path(assets_model_dir, assets_root)
        else:
            assets_model_dir = str(assets_root / "models")
        from pia.utils.api.hugging_face import HFModelDownloader

        downloader = HFModelDownloader()
        model_dir = downloader.download(repo_id=hf_repo_id, save_dir=assets_model_dir, snapshot=True)
        return model_dir

    def _ensure_vit_engine(
        self,
        model_dir: str,
        onnx_file: str,
        trt_file: str,
        image_path: str,
        trt_build: dict,
    ) -> None:
        """VIT TRT 엔진이 없으면 ONNX/엔진을 생성한다.

        Args:
            model_dir: 모델 디렉터리.
            onnx_file: VIT ONNX 파일 경로.
            trt_file: VIT TRT 엔진 파일 경로.
            image_path: VIT 빌드 검증용 이미지 경로.
            trt_build: TRT 빌드 설정.

        Raises:
            RuntimeError: 엔진 빌드 실패 시.
        """
        if trt_file and Path(trt_file).is_file():
            return
        if not image_path or not Path(image_path).is_file():
            image_path = self._select_fallback_image(trt_build)
        from pia.ai.exports.internvl3_trt_convert import build_vit_engine

        build_vit_engine(
            pretrained_model_path=model_dir,
            onnx_file=onnx_file,
            trt_file=trt_file,
            image_path=image_path,
            dtype=trt_build.get("vit_dtype", "bfloat16"),
            min_bs=int(trt_build.get("vit_min_bs", 1)),
            opt_bs=int(trt_build.get("vit_opt_bs", 13)),
            max_bs=int(trt_build.get("vit_max_bs", 26)),
        )

    def _ensure_llm_checkpoint(self, model_dir: str, checkpoint_dir: str, trt_build: dict) -> None:
        """LLM 체크포인트가 없으면 변환을 수행한다.

        Args:
            model_dir: HF 모델 디렉터리.
            checkpoint_dir: 출력 체크포인트 디렉터리.
            trt_build: TRT 빌드 설정.

        Raises:
            RuntimeError: 체크포인트 변환 실패 시.
        """
        if self._is_tllm_checkpoint_ready(checkpoint_dir):
            return
        from pia.ai.exports.internvl3_trt_convert import convert_llm

        convert_llm(
            model_dir=model_dir,
            output_dir=checkpoint_dir,
            dtype=trt_build.get("llm_dtype", "bfloat16"),
            load_model_on_cpu=bool(trt_build.get("load_model_on_cpu", True)),
            disable_rope_scaling=bool(trt_build.get("disable_rope_scaling", True)),
        )

    def _ensure_llm_engine(self, checkpoint_dir: str, output_dir: str, trt_build: dict) -> None:
        """TRT-LLM 엔진이 없으면 빌드를 수행한다.

        Args:
            checkpoint_dir: TRT-LLM 체크포인트 디렉터리.
            output_dir: TRT 엔진 출력 디렉터리.
            trt_build: TRT 빌드 설정.

        Raises:
            RuntimeError: 엔진 빌드 실패 시.
        """
        if self._is_trt_engine_ready(output_dir):
            return
        from pia.ai.exports.internvl3_trt_convert import build_llm_engine

        build_llm_engine(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            max_batch_size=int(trt_build.get("llm_max_batch_size", 4)),
            max_input_len=int(trt_build.get("llm_max_input_len", 3584)),
            max_seq_len=int(trt_build.get("llm_max_seq_len", 3840)),
            max_num_tokens=int(trt_build.get("llm_max_num_tokens", 15200)),
            gemm_plugin=trt_build.get("llm_gemm_plugin", "bfloat16"),
            max_prompt_embedding_table_size=int(
                trt_build.get("llm_max_multimodal_len", 14000)
            ),
            max_multimodal_len=int(trt_build.get("llm_max_multimodal_len", 14000)),
        )

    @staticmethod
    def _is_tllm_checkpoint_ready(checkpoint_dir: Optional[str]) -> bool:
        """TRT-LLM 체크포인트가 준비되었는지 확인한다.

        Args:
            checkpoint_dir: 체크포인트 디렉터리 경로.

        Returns:
            체크포인트 파일이 모두 존재하면 True.
        """
        if not checkpoint_dir:
            return False
        if not Path(checkpoint_dir).is_dir():
            return False
        config_file = Path(checkpoint_dir) / "config.json"
        weight_file = Path(checkpoint_dir) / "rank0.safetensors"
        return config_file.is_file() and weight_file.is_file()

    @staticmethod
    def _is_trt_engine_ready(engine_dir: Optional[str]) -> bool:
        """TRT 엔진이 준비되었는지 확인한다.

        Args:
            engine_dir: 엔진 디렉터리 경로.

        Returns:
            엔진 파일이 존재하면 True.
        """
        if not engine_dir:
            return False
        if not Path(engine_dir).is_dir():
            return False
        config_file = Path(engine_dir) / "config.json"
        if not config_file.is_file():
            return False
        for p in Path(engine_dir).iterdir():
            if p.suffix == ".engine":
                return True
        return False

    def _select_fallback_image(self, trt_build: dict) -> str:
        """샘플 이미지가 없을 때 assets/images에서 대체 이미지를 선택한다.

        Args:
            trt_build: TRT 빌드 설정.

        Returns:
            이미지 경로 문자열.

        Raises:
            FileNotFoundError: 대체 이미지가 없는 경우.
        """
        assets_image_dir = trt_build.get("assets_image_dir", "assets/images")
        assets_image_dir = self._resolve_path(assets_image_dir, self.assets_root)
        if assets_image_dir and Path(assets_image_dir).is_dir():
            exts = (".jpg", ".jpeg", ".png")
            for p in Path(assets_image_dir).iterdir():
                if p.suffix.lower() in exts:
                    return str(p)
        raise FileNotFoundError("VIT 엔진 빌드에 사용할 이미지가 없습니다.")


def build_prompt(user_text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """시스템 프롬프트와 사용자 프롬프트를 합쳐 모델 입력을 만든다.

    Args:
        user_text: 사용자 입력 텍스트. `<image>` 태그가 없으면 앞에 추가한다.
        system_prompt: 시스템 지시문 텍스트.

    Returns:
        InternVL 대화 포맷의 프롬프트 문자열.
    """
    if IMAGE_TAG not in user_text:
        user_text = f"{IMAGE_TAG}\n{user_text}"
    prompt = "<|im_start|>system\n"
    prompt += system_prompt
    prompt += "<|im_end|>\n"
    prompt += "<|im_start|>user\n"
    prompt += user_text
    prompt += "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def build_transform(input_size: int, mean: Sequence[float], std: Sequence[float]):
    """이미지 전처리 변환 파이프라인을 생성한다.

    Args:
        input_size: 리사이즈 크기(정사각).
        mean: 정규화 평균값(BGR/정확히는 RGB로 변환 후 적용).
        std: 정규화 표준편차.

    Returns:
        torchvision.transforms.Compose 객체.
    """
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """가장 가까운 종횡비를 선택한다.

    Args:
        aspect_ratio: 원본 이미지 종횡비(가로/세로).
        target_ratios: 후보 종횡비 목록 (w, h) 튜플.
        width: 원본 이미지 너비.
        height: 원본 이미지 높이.
        image_size: 타일 크기(정사각).

    Returns:
        선택된 (w, h) 종횡비 튜플.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tgt_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tgt_ar)
        if diff < best_ratio_diff or (
            diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """이미지를 타일로 분할하는 동적 전처리를 수행한다.

    Args:
        image: PIL.Image 객체.
        min_num: 최소 타일 수.
        max_num: 최대 타일 수.
        image_size: 타일 한 변 크기.
        use_thumbnail: 여러 타일일 때 썸네일 타일을 추가할지 여부.

    Returns:
        PIL.Image 타일 리스트.
    """
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, ow, oh, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th), resample=Image.BICUBIC)
    tiles = []
    for idx in range(blocks):
        x = (idx % (tw // image_size)) * image_size
        y = (idx // (tw // image_size)) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size), resample=Image.BICUBIC))
    return tiles


def replace_image_tokens(prompt, patch_counts, img_token):
    """프롬프트의 `<image>` 태그를 이미지 토큰으로 확장한다.

    Args:
        prompt: 원본 프롬프트 문자열.
        patch_counts: 이미지별 타일 개수 리스트.
        img_token: 이미지 토큰 문자열.

    Returns:
        `<img>...</img>` 형태로 치환된 프롬프트.

    Raises:
        ValueError: `<image>` 태그 수가 부족한 경우.
    """
    pos = 0
    for patches in patch_counts:
        replace = "<img>" + (img_token * (patches * 256)) + "</img>"
        idx = prompt.find(IMAGE_TAG, pos)
        if idx < 0:
            raise ValueError("Not enough <image> tokens for provided images.")
        prompt = prompt[:idx] + replace + prompt[idx + len(IMAGE_TAG) :]
        pos = idx + len(replace)
    return prompt


def encode_with_img_tokens(tokenizer, text, img_token, img_begin_token_id):
    """이미지 토큰을 포함한 텍스트를 토큰 ID 시퀀스로 변환한다.

    Args:
        tokenizer: HuggingFace tokenizer 객체.
        text: 입력 텍스트.
        img_token: 이미지 토큰 문자열.
        img_begin_token_id: 이미지 토큰 시작 ID.

    Returns:
        토큰 ID 리스트.
    """
    if not img_token:
        return tokenizer.encode(text, add_special_tokens=False)
    parts = text.split(img_token)
    ids = []
    cur_img_id = int(img_begin_token_id)
    for i, part in enumerate(parts):
        if part:
            ids.extend(tokenizer.encode(part, add_special_tokens=False))
        if i < len(parts) - 1:
            ids.append(cur_img_id)
            cur_img_id += 1
    return ids


def decode_tokens(tokenizer, token_ids, skip_ids, img_begin_token_id):
    """토큰 ID를 텍스트로 디코딩한다(이미지 토큰 제외).

    Args:
        tokenizer: HuggingFace tokenizer 객체.
        token_ids: 토큰 ID 리스트.
        skip_ids: 건너뛸 토큰 ID 집합.
        img_begin_token_id: 이미지 토큰 시작 ID.

    Returns:
        디코딩된 문자열.
    """
    filtered = []
    for tid in token_ids:
        if img_begin_token_id and tid >= img_begin_token_id:
            continue
        if tid in skip_ids:
            continue
        filtered.append(tid)
    return tokenizer.decode(filtered, skip_special_tokens=True)


def normalize_output_ids(output_ids):
    """출력 토큰을 1차원 리스트로 정규화한다.

    Args:
        output_ids: torch.Tensor 또는 리스트 중첩 구조.

    Returns:
        1차원 토큰 ID 리스트.
    """
    if torch.is_tensor(output_ids):
        output_ids = output_ids[0].tolist() if output_ids.ndim > 1 else output_ids.tolist()
    while isinstance(output_ids, list) and output_ids and isinstance(output_ids[0], list):
        output_ids = output_ids[0]
    return output_ids


def normalize_output_ids_batch(output_ids):
    """
    배치 출력 토큰을 List[List[int]] 형태로 정규화한다.

    Args:
        output_ids: torch.Tensor/np.ndarray/리스트 형태의 배치 출력.

    Returns:
        배치별 토큰 ID 리스트(List[List[int]]).
    """
    if torch.is_tensor(output_ids):
        output_ids = output_ids.tolist()
    if isinstance(output_ids, np.ndarray):
        output_ids = output_ids.tolist()
    if not isinstance(output_ids, list):
        return [output_ids]
    if output_ids and isinstance(output_ids[0], int):
        return [output_ids]
    normalized = []
    for item in output_ids:
        if torch.is_tensor(item):
            item = item.tolist()
        if isinstance(item, np.ndarray):
            item = item.tolist()
        while isinstance(item, list) and item and isinstance(item[0], list):
            item = item[0]
        normalized.append(item)
    return normalized


def strip_input_prefix(output_ids: List[int], input_ids: List[int]) -> List[int]:
    """
    출력 토큰에 입력 프롬프트 토큰이 포함된 경우 제거한다.
    C++ runner는 exclude_input_from_output을 지원하지 않아 수동으로 처리한다.

    Args:
        output_ids: 모델 출력 토큰 ID 리스트.
        input_ids: 입력 프롬프트 토큰 ID 리스트.

    Returns:
        입력 프롬프트를 제거한 토큰 ID 리스트.
    """
    if not output_ids or not input_ids:
        return output_ids
    in_len = len(input_ids)
    if len(output_ids) >= in_len and output_ids[:in_len] == input_ids:
        return output_ids[in_len:]
    if len(output_ids) >= in_len + 1 and output_ids[1 : 1 + in_len] == input_ids:
        return output_ids[1 + in_len :]
    return output_ids


def trim_trailing_eos_pad(token_ids: List[int], eos_id: Optional[int], pad_id: Optional[int]):
    """출력 끝의 EOS/PAD 토큰을 제거한다.

    Args:
        token_ids: 토큰 ID 리스트.
        eos_id: EOS 토큰 ID.
        pad_id: PAD 토큰 ID.

    Returns:
        EOS/PAD가 제거된 토큰 ID 리스트.
    """
    if (eos_id is None) and (pad_id is None):
        return token_ids
    cut = len(token_ids)
    while cut > 0:
        last = token_ids[cut - 1]
        if (eos_id is not None and last == eos_id) or (pad_id is not None and last == pad_id):
            cut -= 1
            continue
        break
    return token_ids[:cut]


def hash_image_bytes(img: np.ndarray) -> int:
    """
    이미지 바이트 해시를 생성한다. 가능하면 CityHash64를 사용하고,
    실패하면 MD5 기반으로 대체한다.

    Args:
        img: HWC numpy 배열.

    Returns:
        64-bit 정수 해시 값.
    """
    data = img.tobytes()
    try:
        from multimodal_test import cityhash64 as _cityhash64  # local helper

        return int(_cityhash64(data)) & 0xFFFFFFFFFFFFFFFF
    except Exception:
        import hashlib

        digest = hashlib.md5(data).digest()
        return int.from_bytes(digest[:8], "little", signed=False)


def build_extra_ids(input_ids: List[int], img_begin_token_id: int, img_hash: int):
    """이미지 토큰 위치에만 해시 값을 채운 extra_ids를 만든다.

    Args:
        input_ids: 입력 토큰 ID 리스트.
        img_begin_token_id: 이미지 토큰 시작 ID.
        img_hash: 이미지 해시 값.

    Returns:
        extra_ids 리스트 또는 None.
    """
    if not img_begin_token_id:
        return None
    extra_ids = [0] * len(input_ids)
    for i, tid in enumerate(input_ids):
        if tid >= img_begin_token_id:
            extra_ids[i] = img_hash
    return extra_ids


def trt_dtype_to_torch(dtype):
    """TensorRT dtype을 torch dtype으로 변환한다.

    Args:
        dtype: tensorrt.DataType 값.

    Returns:
        torch.dtype 값.

    Raises:
        ValueError: 지원하지 않는 dtype인 경우.
    """
    import tensorrt as trt

    if dtype == trt.DataType.HALF:
        return torch.float16
    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return torch.bfloat16
    if hasattr(trt.DataType, "BFLOAT16") and dtype == trt.DataType.BFLOAT16:
        return torch.bfloat16
    if "BF16" in str(dtype):
        return torch.bfloat16
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    raise ValueError(f"Unsupported TRT dtype: {dtype}")


class TrtVitRunner:
    def __init__(self, engine_path: str):
        """TRT VIT 엔진을 로드하고 실행 컨텍스트를 초기화한다.

        Args:
            engine_path: TensorRT 엔진 파일 경로.

        Raises:
            RuntimeError: 엔진 로딩 또는 I/O 텐서 식별 실패 시.
        """
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inp_name = None
        self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                self.inp_name = name
            else:
                self.out_name = name
        if not self.inp_name or not self.out_name:
            raise RuntimeError("Failed to find TRT input/output tensor names.")
        self.inp_dtype = self.engine.get_tensor_dtype(self.inp_name)
        self.out_dtype = self.engine.get_tensor_dtype(self.out_name)

    def infer(self, inp_tensor: torch.Tensor) -> torch.Tensor:
        """VIT 엔진으로 추론을 수행한다.

        Args:
            inp_tensor: CUDA 상의 입력 텐서 (N, C, H, W).

        Returns:
            VIT 출력 텐서.

        Raises:
            ValueError: 입력 텐서가 CUDA에 없을 때.
        """
        import tensorrt as trt

        if not inp_tensor.is_cuda:
            raise ValueError("Input tensor must be on CUDA.")
        self.context.set_input_shape(self.inp_name, tuple(inp_tensor.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        out_tensor = torch.empty(
            out_shape, device=inp_tensor.device, dtype=trt_dtype_to_torch(self.out_dtype)
        )
        self.context.set_tensor_address(self.inp_name, int(inp_tensor.data_ptr()))
        self.context.set_tensor_address(self.out_name, int(out_tensor.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        return out_tensor


def get_trt_executor():
    """tensorrt_llm executor 모듈을 로드한다.

    Returns:
        executor 모듈.

    Raises:
        RuntimeError: executor 바인딩을 찾지 못한 경우.
    """
    try:
        from tensorrt_llm.bindings import executor as trt_executor
        return trt_executor
    except Exception:
        try:
            import tensorrt_llm.executor as trt_executor
            return trt_executor
        except Exception as exc:
            raise RuntimeError("tensorrt_llm executor bindings not found.") from exc


def make_trt_tensor(trt_executor, torch_tensor: torch.Tensor):
    """torch.Tensor를 TRT-LLM Tensor로 변환한다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        torch_tensor: 변환할 torch 텐서.

    Returns:
        TRT-LLM Tensor 객체 또는 fallback으로 torch tensor.
    """
    torch_tensor = torch_tensor.contiguous()
    tensor_cls = getattr(trt_executor, "Tensor", None)
    if tensor_cls is None:
        for mod_name in ["tensorrt_llm.bindings", "tensorrt_llm.bindings.executor", "tensorrt_llm.executor"]:
            try:
                mod = __import__(mod_name, fromlist=["*"])
            except Exception:
                continue
            tensor_cls = getattr(mod, "Tensor", None)
            if tensor_cls is not None:
                break
    if tensor_cls is None:
        for name in ["as_tensor", "to_tensor", "from_torch", "from_dlpack"]:
            if hasattr(trt_executor, name):
                try:
                    return getattr(trt_executor, name)(torch_tensor)
                except Exception:
                    pass
        return torch_tensor
    if hasattr(tensor_cls, "from_torch"):
        return tensor_cls.from_torch(torch_tensor)
    if hasattr(tensor_cls, "from_dlpack"):
        return tensor_cls.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    if hasattr(tensor_cls, "fromDlpack"):
        return tensor_cls.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    return tensor_cls(torch_tensor)


def make_prompt_tuning_config(trt_executor, ptuning_tensor, extra_ids=None):
    """PromptTuningConfig를 생성한다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        ptuning_tensor: prompt tuning 텐서.
        extra_ids: optional extra id 리스트/텐서.

    Returns:
        PromptTuningConfig 객체.

    Raises:
        RuntimeError: PromptTuningConfig 클래스를 찾지 못한 경우.
    """
    ptuning_cls = getattr(trt_executor, "PromptTuningConfig", None)
    if ptuning_cls is None:
        raise RuntimeError("PromptTuningConfig class not found.")
    try:
        if extra_ids is None:
            return ptuning_cls(ptuning_tensor, None)
        return ptuning_cls(ptuning_tensor, extra_ids)
    except TypeError:
        if extra_ids is None:
            return ptuning_cls(ptuning_tensor)
        return ptuning_cls(ptuning_tensor, extra_ids)


def make_sampling_config(trt_executor, sampling):
    """샘플링 설정 객체를 생성하고 값을 채운다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        sampling: 샘플링 파라미터 dict.

    Returns:
        SamplingConfig 객체 또는 None.
    """
    cfg_cls = getattr(trt_executor, "SamplingConfig", None)
    if cfg_cls is None:
        return None
    cfg = cfg_cls()
    for key, val in sampling.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            setter = f"set{key[0].upper()}{key[1:]}"
            if hasattr(cfg, setter):
                getattr(cfg, setter)(val)
    return cfg


def make_output_config(trt_executor, exclude_input_from_output=True):
    """출력 설정 객체를 생성한다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        exclude_input_from_output: 입력 토큰을 출력에서 제외할지 여부.

    Returns:
        OutputConfig 객체 또는 None.
    """
    cfg_cls = getattr(trt_executor, "OutputConfig", None)
    if cfg_cls is None:
        return None
    cfg = cfg_cls()
    if hasattr(cfg, "excludeInputFromOutput"):
        cfg.excludeInputFromOutput = exclude_input_from_output
    elif hasattr(cfg, "exclude_input_from_output"):
        cfg.exclude_input_from_output = exclude_input_from_output
    return cfg


def get_model_type(trt_executor):
    """executor의 모델 타입(DecoderOnly 등)을 찾는다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.

    Returns:
        모델 타입 enum 값 또는 None.
    """
    model_type = getattr(trt_executor, "ModelType", None)
    if model_type is None:
        return None
    for name in ["kDECODER_ONLY", "DECODER_ONLY", "DecoderOnly"]:
        if hasattr(model_type, name):
            return getattr(model_type, name)
    return None


def get_batching_type(trt_executor):
    """executor의 배칭 타입(InFlight 등)을 찾는다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.

    Returns:
        배칭 타입 enum 값 또는 None.
    """
    batching = getattr(trt_executor, "BatchingType", None)
    if batching is None:
        return None
    for name in ["kINFLIGHT", "INFLIGHT", "InFlight"]:
        if hasattr(batching, name):
            return getattr(batching, name)
    return None


def get_batching_type_from_infer_args(trt_executor, infer_args):
    """infer_args에서 배칭 타입을 추론한다."""
    batching = getattr(trt_executor, "BatchingType", None)
    if batching is None:
        return None

    gpt_model_type = ""
    if infer_args:
        gpt_model_type = str(infer_args.get("gpt_model_type", "")).lower()

    if gpt_model_type in {"v1", "static"}:
        for name in ["kSTATIC", "STATIC", "Static"]:
            if hasattr(batching, name):
                return getattr(batching, name)
    if gpt_model_type in {"inflight_batching", "inflight_fused_batching"}:
        for name in ["kINFLIGHT", "INFLIGHT", "InFlight"]:
            if hasattr(batching, name):
                return getattr(batching, name)

    for name in ["kINFLIGHT", "INFLIGHT", "InFlight"]:
        if hasattr(batching, name):
            return getattr(batching, name)
    for name in ["kSTATIC", "STATIC", "Static"]:
        if hasattr(batching, name):
            return getattr(batching, name)
    return None


def apply_kv_cache_overrides(cfg, infer_args, enable_kv_cache_reuse=None):
    """KV 캐시 관련 설정을 infer_args로 덮어쓴다.

    Args:
        cfg: ExecutorConfig 객체.
        infer_args: 추론 설정 dict.
    """
    if not infer_args:
        return
    kv_cfg = getattr(cfg, "kv_cache_config", None)
    if kv_cfg is None:
        return
    max_tokens = infer_args.get("max_tokens_in_paged_kv_cache")
    if max_tokens is not None:
        try:
            kv_cfg.max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            pass
    free_fraction = infer_args.get("kv_cache_free_gpu_mem_fraction")
    if free_fraction is not None:
        try:
            kv_cfg.free_gpu_memory_fraction = float(free_fraction)
        except (TypeError, ValueError):
            pass
    if enable_kv_cache_reuse is not None:
        try:
            if hasattr(kv_cfg, "enable_kv_cache_reuse"):
                kv_cfg.enable_kv_cache_reuse = bool(enable_kv_cache_reuse)
            elif hasattr(kv_cfg, "setEnableKvCacheReuse"):
                kv_cfg.setEnableKvCacheReuse(bool(enable_kv_cache_reuse))
        except Exception:
            pass


def make_executor_config(trt_executor, infer_args=None, enable_kv_cache_reuse=None):
    """ExecutorConfig를 생성하고 배칭/캐시 설정을 적용한다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        infer_args: 추론 설정 dict.

    Returns:
        ExecutorConfig 객체 또는 None.
    """
    cfg_cls = getattr(trt_executor, "ExecutorConfig", None)
    if cfg_cls is None:
        return None
    cfg = cfg_cls()
    batching_type = get_batching_type_from_infer_args(trt_executor, infer_args)
    if batching_type is None:
        batching_type = get_batching_type(trt_executor)
    if batching_type is not None:
        if hasattr(cfg, "batching_type"):
            cfg.batching_type = batching_type
        elif hasattr(cfg, "setBatchingType"):
            cfg.setBatchingType(batching_type)
    apply_kv_cache_overrides(cfg, infer_args, enable_kv_cache_reuse=enable_kv_cache_reuse)
    return cfg


def create_executor(trt_executor, engine_dir, infer_args=None, enable_kv_cache_reuse=None):
    """TRT-LLM Executor를 생성한다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        engine_dir: 엔진 디렉터리 경로.
        infer_args: 추론 설정 dict.

    Returns:
        Executor 인스턴스.
    """
    model_type = get_model_type(trt_executor)
    cfg = make_executor_config(trt_executor, infer_args, enable_kv_cache_reuse=enable_kv_cache_reuse)
    if cfg is not None and model_type is not None:
        return trt_executor.Executor(engine_dir, model_type, cfg)
    if model_type is not None:
        return trt_executor.Executor(engine_dir, model_type)
    return trt_executor.Executor(engine_dir)


def create_cpp_runner(engine_dir, infer_args):
    """C++ ModelRunner를 생성한다.

    Args:
        engine_dir: 엔진 디렉터리 경로.
        infer_args: 추론 설정 dict.

    Returns:
        ModelRunnerCpp 인스턴스.

    Raises:
        RuntimeError: tensorrt_llm.runtime 로딩 실패 시.
    """
    try:
        from tensorrt_llm.runtime import ModelRunnerCpp
    except Exception as exc:
        raise RuntimeError("tensorrt_llm.runtime is required for C++ runner.") from exc
    kwargs = {}
    if infer_args:
        kv_fraction = infer_args.get("kv_cache_free_gpu_mem_fraction")
        if kv_fraction is not None:
            kwargs["kv_cache_free_gpu_memory_fraction"] = float(kv_fraction)
        max_tokens = infer_args.get("max_tokens_in_paged_kv_cache")
        if max_tokens is not None:
            kwargs["max_tokens_in_paged_kv_cache"] = int(max_tokens)
        enable_chunked_context = infer_args.get("enable_chunked_context")
        if enable_chunked_context is not None:
            kwargs["enable_chunked_context"] = bool(enable_chunked_context)
    return ModelRunnerCpp.from_dir(engine_dir, rank=0, debug_mode=False, **kwargs)


def create_request(
    trt_executor,
    input_ids,
    max_new_tokens,
    sampling_config,
    output_config,
    ptuning_config,
    end_id,
    pad_id,
):
    """TRT-LLM Request 객체를 생성한다.

    Args:
        trt_executor: tensorrt_llm executor 모듈.
        input_ids: 입력 토큰 ID 리스트.
        max_new_tokens: 생성 최대 토큰 수.
        sampling_config: SamplingConfig 객체.
        output_config: OutputConfig 객체.
        ptuning_config: PromptTuningConfig 객체.
        end_id: EOS 토큰 ID.
        pad_id: PAD 토큰 ID.

    Returns:
        Request 객체.

    Raises:
        RuntimeError: Request 클래스를 찾지 못한 경우.
    """
    req_cls = getattr(trt_executor, "Request", None)
    if req_cls is None:
        raise RuntimeError("Request class not found.")
    base_kwargs = {
        "streaming": False,
        "sampling_config": sampling_config,
        "output_config": output_config,
        "end_id": end_id,
        "pad_id": pad_id,
        "prompt_tuning_config": ptuning_config,
    }
    for input_key in ["input_token_ids", "input_ids"]:
        for max_key in ["max_tokens", "max_new_tokens"]:
            kwargs = dict(base_kwargs)
            kwargs[input_key] = input_ids
            kwargs[max_key] = max_new_tokens
            try:
                return req_cls(**kwargs)
            except TypeError:
                continue
    return req_cls(
        input_ids,
        max_new_tokens,
        False,
        sampling_config,
        output_config,
        end_id,
        pad_id,
        None,
        None,
        None,
        None,
        None,
        ptuning_config,
    )


def enqueue_request(executor, request):
    """Executor에 request를 enqueue한다.

    Args:
        executor: TRT-LLM Executor.
        request: Request 객체.

    Returns:
        request_id 또는 요청 핸들.

    Raises:
        RuntimeError: enqueue 메서드를 찾지 못한 경우.
    """
    for name in ["enqueue_request", "enqueueRequest", "enqueue_requests", "enqueueRequests"]:
        if hasattr(executor, name):
            fn = getattr(executor, name)
            try:
                return fn(request)
            except TypeError:
                return fn([request])
    raise RuntimeError("No enqueue method found on Executor.")


def _timeout_arg(wait_ms):
    if wait_ms is None:
        return None
    try:
        return datetime.timedelta(milliseconds=int(wait_ms))
    except Exception:
        return None


def await_responses(executor, wait_ms):
    """응답을 대기하여 수신한다.

    Args:
        executor: TRT-LLM Executor.
        wait_ms: 대기 시간(ms).

    Returns:
        응답 리스트 또는 None.

    Raises:
        RuntimeError: awaitResponses 메서드를 찾지 못한 경우.
    """
    for name in ["await_responses", "awaitResponses"]:
        if hasattr(executor, name):
            fn = getattr(executor, name)
            try:
                return fn(timeout=_timeout_arg(wait_ms))
            except TypeError:
                try:
                    return fn(wait_ms)
                except TypeError:
                    return fn()
    raise RuntimeError("No awaitResponses method found on Executor.")


def get_num_responses_ready(executor):
    """현재 준비된 응답 개수를 조회한다.

    Args:
        executor: TRT-LLM Executor.

    Returns:
        준비된 응답 개수 또는 None.
    """
    for name in ["get_num_responses_ready", "getNumResponsesReady"]:
        if hasattr(executor, name):
            return getattr(executor, name)()
    return None


def normalize_request_id(request_id):
    """request_id 반환 형식을 스칼라로 정규화한다.

    Args:
        request_id: 단일/리스트/ndarray 형태의 request_id.

    Returns:
        정규화된 request_id 또는 None.
    """
    if isinstance(request_id, (list, tuple)):
        return request_id[0] if request_id else None
    if isinstance(request_id, np.ndarray):
        if request_id.size:
            return request_id.flat[0].item()
        return None
    return request_id


def get_response_request_id(resp):
    """응답 객체에서 request_id를 추출한다.

    Args:
        resp: 응답 객체.

    Returns:
        request_id 또는 None.
    """
    for name in ["request_id", "requestId", "getRequestId", "get_request_id"]:
        val, seen = _read_attr(resp, name)
        if seen and val is not None:
            return val
    return None


def is_final_response(resp):
    """응답이 최종 응답인지 판별한다.

    Args:
        resp: 응답 객체.

    Returns:
        최종 응답이면 True.
    """
    for name in ["is_final", "isFinal", "get_is_final", "getIsFinal"]:
        val, seen = _read_attr(resp, name)
        if seen and val is not None:
            return bool(val)
    for name in ["getResult", "result", "get_result"]:
        result, seen = _read_attr(resp, name)
        if seen and result is not None:
            for rname in ["isFinal", "is_final", "getIsFinal", "get_is_final"]:
                val, rseen = _read_attr(result, rname)
                if rseen and val is not None:
                    try:
                        return bool(val)
                    except Exception:
                        pass
    return True


def _read_attr(obj, name):
    try:
        attr = getattr(obj, name)
    except Exception:
        return None, False
    try:
        return (attr() if callable(attr) else attr), True
    except Exception:
        return None, True


def _try_output_ids(obj):
    for name in [
        "output_token_ids",
        "outputTokenIds",
        "getOutputTokenIds",
        "get_output_token_ids",
        "output_ids",
        "outputIds",
        "getOutputIds",
        "get_output_ids",
    ]:
        val, seen = _read_attr(obj, name)
        if seen and val is not None:
            return val
    return None


def extract_output_token_ids(resp):
    """응답 객체에서 출력 토큰 ID를 추출한다.

    Args:
        resp: 응답 객체.

    Returns:
        출력 토큰 ID.

    Raises:
        RuntimeError: 출력 토큰 필드를 찾지 못한 경우.
    """
    if isinstance(resp, dict):
        for key in ["output_token_ids", "outputTokenIds", "output_ids", "outputIds"]:
            if key in resp:
                return resp[key]

    out = _try_output_ids(resp)
    if out is not None:
        return out

    for name in ["getResult", "result", "get_result"]:
        if hasattr(resp, name):
            try:
                result = getattr(resp, name)()
            except TypeError:
                result = getattr(resp, name)
            except Exception:
                result = None
            if result is not None:
                out = _try_output_ids(result)
                if out is not None:
                    return out

    err_msg = None
    for name in ["getErrorMsg", "errorMsg", "get_error_msg"]:
        val, _ = _read_attr(resp, name)
        if val:
            err_msg = val
            break
    if err_msg is None and hasattr(resp, "getResult"):
        try:
            result = resp.getResult()
        except Exception:
            result = None
        if result is not None:
            for name in ["getErrorMsg", "errorMsg", "get_error_msg"]:
                val, _ = _read_attr(result, name)
                if val:
                    err_msg = val
                    break

    details = f"type={type(resp)}"
    if hasattr(resp, "__dict__"):
        details += f", attrs={list(resp.__dict__.keys())}"
    raise RuntimeError(f"No output token ids found in response. {details} err={err_msg}")


def wait_for_response(executor, request_id, timeout_ms=60000):
    """단일 request_id에 대한 최종 응답을 대기한다.

    Args:
        executor: TRT-LLM Executor.
        request_id: 요청 ID.
        timeout_ms: 타임아웃(ms).

    Returns:
        응답 객체.

    Raises:
        RuntimeError: 타임아웃 또는 응답 수신 실패 시.
    """
    start = time.time()
    last = None
    request_id = normalize_request_id(request_id)
    while (time.time() - start) * 1000 < timeout_ms:
        ready = get_num_responses_ready(executor)
        if ready is not None and ready <= 0:
            time.sleep(0.01)
            continue
        responses = await_responses(executor, 0 if ready is not None else 10)
        if responses is None:
            time.sleep(0.01)
            continue
        for resp in responses:
            rid = get_response_request_id(resp)
            if request_id is None or rid == request_id:
                last = resp
                if is_final_response(resp):
                    return resp
        time.sleep(0.01)
    if last is not None:
        return last
    raise RuntimeError("Timed out waiting for response.")


def wait_for_all_responses(executor, request_ids, timeout_ms=60000):
    """
    여러 request_id에 대한 응답을 동기적으로 수집한다.
    모든 응답이 도착하면 요청 순서대로 반환한다.

    Args:
        executor: TRT-LLM Executor.
        request_ids: 요청 ID 리스트.
        timeout_ms: 타임아웃(ms).

    Returns:
        요청 순서대로 정렬된 응답 리스트.

    Raises:
        RuntimeError: 타임아웃 시.
    """
    results = {}
    pending = set(request_ids)
    start = time.time()
    while pending and (time.time() - start) * 1000 < timeout_ms:
        ready = get_num_responses_ready(executor)
        if ready is not None and ready <= 0:
            time.sleep(0.01)
            continue
        responses = await_responses(executor, 0 if ready is not None else 10)
        if responses is None:
            time.sleep(0.01)
            continue
        for resp in responses:
            rid = get_response_request_id(resp)
            if rid in pending and is_final_response(resp):
                results[rid] = resp
                pending.discard(rid)
        time.sleep(0.01)
    if pending:
        raise RuntimeError(f"Timed out waiting for {len(pending)} responses.")
    return [results[rid] for rid in request_ids]
