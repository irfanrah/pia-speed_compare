"""
TensorRT-LLM engine builder wrapper.

Wraps `trtllm-build` as a Python-callable function.

Usage as module:
    from pia.ai.exports.vqa_trt_convert.convert_llm_build import build_llm_engine

    build_llm_engine(
        checkpoint_dir="/assets/InternVL3-2B/tllm_checkpoint/",
        output_dir="/assets/InternVL3-2B/trt_engines/",
        max_batch_size=2,
        max_input_len=7312,
        max_seq_len=7326,
        max_num_tokens=15200,
        max_prompt_embedding_table_size=4608,
        max_multimodal_len=4608,
        gemm_plugin="bfloat16",
    )

Usage as CLI:
    python -m pia.ai.exports.vqa_trt_convert.convert_llm_build \
        --checkpoint_dir /assets/InternVL3-2B/tllm_checkpoint/ \
        --output_dir /assets/InternVL3-2B/trt_engines/ \
        --max_batch_size 2 --max_input_len 7312 --max_seq_len 7326 \
        --max_num_tokens 15200 --max_prompt_embedding_table_size 4608 \
        --gemm_plugin bfloat16 --max_multimodal_len 4608
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import parallel_build
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from tensorrt_llm.plugin import PluginConfig


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class BuildLLMConfig:
    checkpoint_dir: str = ""
    output_dir: str = "trt_engines"

    # build limits
    # 토큰/메모리 산정(대략):
    #   seq_len_i = input_len_i + new_tokens_i <= max_seq_len
    #   batch_tokens = sum(seq_len_i) <= max_num_tokens
    #   균등 배치 가정 시 1요청당 상한 ≈ floor(max_num_tokens / max_batch_size)
    #   KV 캐시 메모리 ≈ max_num_tokens * num_layers * 2(K/V)
    #                    * num_kv_heads * head_dim * bytes_per_elem
    #   paged_kv_cache 사용 시 블록 단위 할당:
    #     alloc_tokens ≈ ceil(max_num_tokens / tokens_per_block) * tokens_per_block
    #     (tokens_per_block: trtllm 기본 32)
    #   (모델 구조/정밀도/플러그인에 따라 실제 값은 달라짐)


    #### max_batch_size, max_num_tokens → 배치 전체 기준(여러 요청 합)
    ## max_input_len, max_seq_len, max_encoder_input_len, max_beam_width → 요청 1개당 기준
    ## 요청이라고 표현하는데 그냥 쓰기로 함

    max_batch_size: int = 2  # 배치당 요청 수 상한(동시 요청 수)
    max_input_len: int = 7312  # 요청 1개당 입력 최대 토큰(텍스트+이미지 토큰 합)
    max_seq_len: int = 7326  # 요청 1개당 입력+출력 최대 토큰(max_new_tokens = max_seq_len - max_input_len)
    max_num_tokens: int = 15200  # 배치 전체 토큰 합 상한(Σ seq_len_i, padding 제거 기준)
    max_beam_width: int = 1  # 요청 1개당 beam search 폭(>1이면 효과적 배치/메모리 증가)
    opt_num_tokens: Optional[int] = None  # 배치 토큰 최적화 타깃(기본값≈max_batch_size*max_beam_width)
    max_prompt_embedding_table_size: int = 4608  # 요청 1개당 프롬프트 임베딩 테이블 최대 길이(이미지 토큰)
    max_encoder_input_len: int = 1024  # 요청 1개당 인코더 입력 길이 상한(비전 인코더 등)

    # plugins
    gemm_plugin: Optional[str] = "bfloat16"  # GEMM 플러그인 정밀도(bfloat16/float16)
    paged_kv_cache: bool = True  # KV 캐시 페이징 사용(긴 시퀀스 메모리 절감)
    use_paged_context_fmha: bool = True  # paged-context FMHA 사용
    context_fmha: bool = True  # context FMHA 사용
    use_fused_mlp: bool = True  # fused MLP 사용(성능)
    remove_input_padding: bool = True  # 입력 padding 제거(성능/메모리)

    # logits
    gather_context_logits: bool = False  # context 단계 logits 저장(메모리/IO 증가)
    gather_generation_logits: bool = False  # generation 단계 logits 저장
    gather_all_token_logits: bool = False  # 모든 토큰 logits 저장(상위 2개 모두 True)

    # misc
    max_multimodal_len: int = 4608  # 멀티모달 토큰 최대 길이(이미지 토큰)
    workers: int = 1  # 병렬 빌드 worker 수
    log_level: str = "info"  # 로그 레벨
    logits_dtype: Optional[str] = None  # logits 출력 dtype(필요 시 강제)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_llm_engine(
    checkpoint_dir: str,
    output_dir: str,
    max_batch_size: int = 2,  # 배치당 요청 수 상한(동시 요청 수)
    max_input_len: int = 7312,  # 요청 1개당 입력 최대 토큰(텍스트+이미지 토큰 합)
    max_seq_len: int = 7326,  # 요청 1개당 입력+출력 최대 토큰(max_new_tokens = max_seq_len - max_input_len)
    max_num_tokens: int = 15200,  # 배치 전체 토큰 합 상한(Σ seq_len_i, padding 제거 기준)
    max_beam_width: int = 1,  # 요청 1개당 beam search 폭(>1이면 효과적 배치/메모리 증가)
    opt_num_tokens: Optional[int] = None,  # 배치 토큰 최적화 타깃(기본값≈max_batch_size*max_beam_width)
    max_prompt_embedding_table_size: int = 4608,  # 요청 1개당 프롬프트 임베딩 테이블 최대 길이(이미지 토큰)
    max_encoder_input_len: int = 1024,  # 요청 1개당 인코더 입력 길이 상한(비전 인코더 등)
    gemm_plugin: Optional[str] = "bfloat16",  # GEMM 플러그인 정밀도(bfloat16/float16)
    paged_kv_cache: bool = True,  # KV 캐시 페이징 사용(긴 시퀀스 메모리 절감)
    use_paged_context_fmha: bool = True,  # paged-context FMHA 사용
    context_fmha: bool = True,  # context FMHA 사용
    use_fused_mlp: bool = True,  # fused MLP 사용(성능)
    remove_input_padding: bool = True,  # 입력 padding 제거(성능/메모리)
    gather_context_logits: bool = False,  # context 단계 logits 저장(메모리/IO 증가)
    gather_generation_logits: bool = False,  # generation 단계 logits 저장
    gather_all_token_logits: bool = False,  # 모든 토큰 logits 저장(상위 2개 모두 True)
    max_multimodal_len: int = 4608,  # 멀티모달 토큰 최대 길이(이미지 토큰)  <- 이거는 뒤에 길이 체크용(빌드할때 안씀)
    workers: int = 1,  # 병렬 빌드 worker 수
    log_level: str = "info",  # 로그 레벨
    logits_dtype: Optional[str] = None,  # logits 출력 dtype(필요 시 강제)
) -> str:
    """Build a TensorRT-LLM engine from a converted checkpoint.

    Equivalent to running:
        trtllm-build --checkpoint_dir ... --output_dir ... [options]

    Returns the output_dir path on success.
    """
    from tensorrt_llm.logger import logger
    logger.set_level(log_level)

    if gather_all_token_logits:
        gather_context_logits = True
        gather_generation_logits = True

    os.makedirs(output_dir, exist_ok=True)

    # ---- load model config from checkpoint --------------------------------
    config_path = os.path.join(checkpoint_dir, "config.json")
    model_config = PretrainedConfig.from_json_file(config_path)

    # ---- plugin config ----------------------------------------------------
    plugin_config = PluginConfig()
    plugin_config.gemm_plugin = gemm_plugin
    plugin_config.context_fmha = context_fmha
    plugin_config.paged_kv_cache = paged_kv_cache
    plugin_config.use_paged_context_fmha = use_paged_context_fmha
    plugin_config.use_fused_mlp = use_fused_mlp
    plugin_config.remove_input_padding = remove_input_padding

    # ---- build config -----------------------------------------------------
    build_dict = {
        "max_input_len": max_input_len,
        "max_seq_len": max_seq_len,
        "max_batch_size": max_batch_size,
        "max_beam_width": max_beam_width,
        "max_num_tokens": max_num_tokens,
        "opt_num_tokens": opt_num_tokens,
        "max_prompt_embedding_table_size": max_prompt_embedding_table_size,
        "gather_context_logits": gather_context_logits,
        "gather_generation_logits": gather_generation_logits,
        "strongly_typed": True,
        "max_encoder_input_len": max_encoder_input_len,
        "use_mrope": (
            True if getattr(model_config, "qwen_type", None) == "qwen2_vl"
            else False
        ),
    }
    build_config = BuildConfig.from_dict(build_dict, plugin_config=plugin_config)

    # ---- kwargs forwarded to build_model ----------------------------------
    kwargs = {
        "logits_dtype": logits_dtype,
        "use_fused_mlp": use_fused_mlp,
        "lora_dir": None,
        "lora_ckpt_source": "hf",
        "max_lora_rank": 64,
        "lora_target_modules": None,
        "strip_plan": False,
        "refit": False,
    }

    effective_workers = min(torch.cuda.device_count(), workers)

    tik = time.time()
    parallel_build(
        model_config,
        checkpoint_dir,
        build_config,
        output_dir,
        effective_workers,
        log_level,
        model_cls=None,
        **kwargs,
    )
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - tik))
    logger.info(f"Total time of building all engines: {elapsed}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TRT-LLM engine from checkpoint (Python wrapper).",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trt_engines")
    parser.add_argument("--max_batch_size", type=int, default=2)
    parser.add_argument("--max_input_len", type=int, default=7312)
    parser.add_argument("--max_seq_len", type=int, default=7326)
    parser.add_argument("--max_num_tokens", type=int, default=15200)
    parser.add_argument("--max_beam_width", type=int, default=1)
    parser.add_argument("--opt_num_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_embedding_table_size", type=int,
                        default=4608)
    parser.add_argument("--max_encoder_input_len", type=int, default=1024)
    parser.add_argument("--gemm_plugin", type=str, default="bfloat16")
    parser.add_argument("--paged_kv_cache", type=str, default="enable",
                        choices=["enable", "disable"])
    parser.add_argument("--use_paged_context_fmha", type=str, default="enable",
                        choices=["enable", "disable"])
    parser.add_argument("--context_fmha", type=str, default="enable",
                        choices=["enable", "disable"])
    parser.add_argument("--use_fused_mlp", type=str, default="enable",
                        choices=["enable", "disable"])
    parser.add_argument("--remove_input_padding", type=str, default="enable",
                        choices=["enable", "disable"])
    parser.add_argument("--gather_context_logits", action="store_true",
                        default=False)
    parser.add_argument("--gather_generation_logits", action="store_true",
                        default=False)
    parser.add_argument("--gather_all_token_logits", action="store_true",
                        default=False)
    parser.add_argument("--max_multimodal_len", type=int, default=4608)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--logits_dtype", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_arguments()
    # convert enable/disable strings to booleans
    bool_fields = [
        "paged_kv_cache", "use_paged_context_fmha", "context_fmha",
        "use_fused_mlp", "remove_input_padding",
    ]
    kwargs = vars(args)
    for f in bool_fields:
        kwargs[f] = kwargs[f] == "enable"
    build_llm_engine(**kwargs)


if __name__ == "__main__":
    main()
