"""
InternVL / Qwen2 LLM checkpoint -> TensorRT-LLM checkpoint converter.

Usage as module:
    from pia.ai.exports.vqa_trt_convert.convert_llm import convert_llm

    convert_llm(
        model_dir="/assets/InternVL3-2B",
        output_dir="/assets/InternVL3-2B/tllm_checkpoint/",
        dtype="bfloat16",
        load_model_on_cpu=True,
        disable_rope_scaling=True,
    )

Usage as CLI:
    python -m pia.ai.exports.vqa_trt_convert.convert_llm \
        --model_dir /assets/InternVL3-2B \
        --output_dir /assets/InternVL3-2B/tllm_checkpoint/ \
        --dtype bfloat16 --load_model_on_cpu --disable_rope_scaling
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.qwen.convert import load_hf_qwen
from tensorrt_llm.quantization import QuantAlgo


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConvertLLMConfig:
    model_dir: str = ""
    output_dir: str = "tllm_checkpoint"
    dtype: str = "float16"  # 가중치/활성화 저장 dtype
    tp_size: int = 1  # Tensor Parallel 크기(가중치/연산 분할)
    pp_size: int = 1  # Pipeline Parallel 단계 수
    workers: int = 1  # 변환 병렬 worker 수(랭크별 변환)

    # quantisation
    use_weight_only: bool = False  # weight-only 양자화 사용 여부
    weight_only_precision: str = "int8"  # int8/int4/int4_gptq
    disable_weight_only_quant_plugin: bool = False  # weight-only 플러그인 비활성화
    smoothquant: Optional[float] = None  # SmoothQuant 계수(예: 0.5)
    per_channel: bool = False  # 채널별 스케일
    per_token: bool = False  # 토큰별 스케일
    int8_kv_cache: bool = False  # KV 캐시를 INT8로 저장(메모리 절감)
    per_group: bool = False  # 그룹 단위 스케일(주로 GPTQ)
    group_size: int = 128  # GPTQ 그룹 크기
    calib_dataset: str = "ccdv/cnn_dailymail"  # SmoothQuant 캘리브레이션 데이터셋

    # embedding
    use_parallel_embedding: bool = False  # 임베딩 테이블을 TP로 분할
    embedding_sharding_dim: int = 0  # 0: vocab 차원 분할, 1: hidden 차원 분할
    use_embedding_sharing: bool = False  # input/output embedding 공유

    # MoE
    moe_tp_size: int = -1  # MoE Tensor Parallel (기본 tp_size)
    moe_ep_size: int = -1  # MoE Expert Parallel (기본 1)

    # misc
    load_model_on_cpu: bool = False  # HF 로드 시 CPU 사용(메모리 절감)
    disable_rope_scaling: bool = False  # Dynamic RoPE scaling 비활성화


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_quant_config(cfg: ConvertLLMConfig) -> QuantConfig:
    quant_config = QuantConfig()
    if cfg.use_weight_only:
        if cfg.weight_only_precision == "int8":
            quant_config.quant_algo = QuantAlgo.W8A16
        elif cfg.weight_only_precision == "int4":
            quant_config.quant_algo = QuantAlgo.W4A16
    elif cfg.smoothquant:
        quant_config.smoothquant_val = cfg.smoothquant
        if cfg.per_channel:
            if cfg.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if cfg.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if cfg.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    if cfg.weight_only_precision == "int4_gptq":
        quant_config.group_size = cfg.group_size
        quant_config.has_zero_point = True
        quant_config.pre_quant_scale = False
        quant_config.quant_algo = QuantAlgo.W4A16_GPTQ

    return quant_config


def _build_options(cfg: ConvertLLMConfig) -> dict:
    return {
        "use_parallel_embedding": cfg.use_parallel_embedding,
        "embedding_sharding_dim": cfg.embedding_sharding_dim,
        "share_embedding_table": cfg.use_embedding_sharing,
        "disable_weight_only_quant_plugin": cfg.disable_weight_only_quant_plugin,
    }


def _execute(workers: int, func_list, cfg: ConvertLLMConfig):
    if workers == 1:
        for rank, f in enumerate(func_list):
            f(cfg, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, cfg, rank) for rank, f in enumerate(func_list)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions) == 0, \
                "Checkpoint conversion failed, please check error log."


def _convert_and_save_hf(cfg: ConvertLLMConfig):
    model_dir = cfg.model_dir
    world_size = cfg.tp_size * cfg.pp_size
    override_fields = _build_options(cfg)

    use_hf_gptq_checkpoint = (
        cfg.use_weight_only and cfg.weight_only_precision == "int4_gptq"
    )
    quant_config = _build_quant_config(cfg)

    if cfg.smoothquant is not None or cfg.int8_kv_cache:
        mapping = Mapping(
            world_size=world_size,
            tp_size=cfg.tp_size,
            pp_size=cfg.pp_size,
            moe_tp_size=cfg.moe_tp_size,
            moe_ep_size=cfg.moe_ep_size,
        )
        QWenForCausalLM.quantize(
            cfg.model_dir,
            cfg.output_dir,
            dtype=cfg.dtype,
            mapping=mapping,
            quant_config=quant_config,
            calib_dataset=cfg.calib_dataset,
            **override_fields,
        )
    else:
        hf_model = load_hf_qwen(model_dir, cfg.load_model_on_cpu).language_model
        os.environ["TRTLLM_DISABLE_UNIFIED_CONVERTER"] = "1"

        disable_rope = cfg.disable_rope_scaling
        if disable_rope:
            print("[INFO] Dynamic RoPE scaling disabled via disable_rope_scaling")

        if hf_model is not None:
            cfgs = [hf_model.config]
            if hasattr(hf_model.config, "llm_config"):
                cfgs.append(hf_model.config.llm_config)
            for hf_cfg in cfgs:
                if disable_rope:
                    if hasattr(hf_cfg, "rope_scaling"):
                        hf_cfg.rope_scaling = None
                else:
                    rope_scaling = getattr(hf_cfg, "rope_scaling", None)
                    if isinstance(rope_scaling, dict):
                        if (rope_scaling.get("alpha") is None
                                and rope_scaling.get("factor") is not None):
                            rope_scaling["alpha"] = rope_scaling["factor"]

        def convert_and_save_rank(cfg_inner, rank):
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=cfg_inner.tp_size,
                pp_size=cfg_inner.pp_size,
                moe_tp_size=cfg_inner.moe_tp_size,
                moe_ep_size=cfg_inner.moe_ep_size,
            )
            qwen = QWenForCausalLM.from_hugging_face(
                model_dir if hf_model is None else hf_model,
                cfg_inner.dtype,
                mapping=mapping,
                quant_config=quant_config,
                use_hf_gptq_checkpoint=use_hf_gptq_checkpoint,
                **override_fields,
            )
            if disable_rope:
                qwen.config.rotary_scaling = None
                if rank == 0:
                    print("[INFO] rotary_scaling set to None (dynamic RoPE disabled)")
            elif qwen.config.rotary_scaling is None:
                src_cfg = hf_model.config if hf_model is not None else None
                if src_cfg is not None and hasattr(src_cfg, "llm_config"):
                    src_cfg = src_cfg.llm_config
                rope_scaling = getattr(src_cfg, "rope_scaling", None)
                if isinstance(rope_scaling, dict):
                    if (rope_scaling.get("alpha") is None
                            and rope_scaling.get("factor") is not None):
                        rope_scaling["alpha"] = rope_scaling["factor"]
                    qwen.config.rotary_scaling = rope_scaling
            qwen.save_checkpoint(cfg_inner.output_dir, save_config=(rank == 0))
            del qwen

        _execute(cfg.workers, [convert_and_save_rank] * world_size, cfg)
        release_gc()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def convert_llm(
    model_dir: str,
    output_dir: str,
    dtype: str = "bfloat16",  # 가중치/활성화 저장 dtype
    load_model_on_cpu: bool = True,  # HF 로드 시 CPU 사용(메모리 절감)
    disable_rope_scaling: bool = False,  # Dynamic RoPE scaling 비활성화
    tp_size: int = 1,  # Tensor Parallel 크기
    pp_size: int = 1,  # Pipeline Parallel 단계 수
    workers: int = 1,  # 변환 병렬 worker 수(랭크별 변환)
    use_weight_only: bool = False,  # weight-only 양자화 사용 여부
    weight_only_precision: str = "int8",  # int8/int4/int4_gptq
    disable_weight_only_quant_plugin: bool = False,  # weight-only 플러그인 비활성화
    smoothquant: Optional[float] = None,  # SmoothQuant 계수(예: 0.5)
    per_channel: bool = False,  # 채널별 스케일
    per_token: bool = False,  # 토큰별 스케일
    int8_kv_cache: bool = False,  # KV 캐시를 INT8로 저장(메모리 절감)
    per_group: bool = False,  # 그룹 단위 스케일(GPTQ)
    group_size: int = 128,  # GPTQ 그룹 크기
    calib_dataset: str = "ccdv/cnn_dailymail",  # SmoothQuant 캘리브레이션 데이터셋
    use_parallel_embedding: bool = False,  # 임베딩 테이블을 TP로 분할
    embedding_sharding_dim: int = 0,  # 0: vocab 차원 분할, 1: hidden 차원 분할
    use_embedding_sharing: bool = False,  # input/output embedding 공유
    moe_tp_size: int = -1,  # MoE Tensor Parallel (기본 tp_size)
    moe_ep_size: int = -1,  # MoE Expert Parallel (기본 1)
) -> str:
    """Convert a HuggingFace Qwen2/InternVL LLM checkpoint to TRT-LLM format.

    Returns the output_dir path on success.
    """
    print(f"tensorrt_llm {tensorrt_llm.__version__}")

    cfg = ConvertLLMConfig(
        model_dir=model_dir,
        output_dir=output_dir,
        dtype=dtype,
        load_model_on_cpu=load_model_on_cpu,
        disable_rope_scaling=disable_rope_scaling,
        tp_size=tp_size,
        pp_size=pp_size,
        workers=workers,
        use_weight_only=use_weight_only,
        weight_only_precision=weight_only_precision,
        disable_weight_only_quant_plugin=disable_weight_only_quant_plugin,
        smoothquant=smoothquant,
        per_channel=per_channel,
        per_token=per_token,
        int8_kv_cache=int8_kv_cache,
        per_group=per_group,
        group_size=group_size,
        calib_dataset=calib_dataset,
        use_parallel_embedding=use_parallel_embedding,
        embedding_sharding_dim=embedding_sharding_dim,
        use_embedding_sharing=use_embedding_sharing,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
    )

    # resolve MoE defaults
    if cfg.moe_tp_size == -1 and cfg.moe_ep_size == -1:
        cfg.moe_tp_size = cfg.tp_size
        cfg.moe_ep_size = 1
    elif cfg.moe_tp_size == -1:
        cfg.moe_tp_size = cfg.tp_size // cfg.moe_ep_size
    elif cfg.moe_ep_size == -1:
        cfg.moe_ep_size = cfg.tp_size // cfg.moe_tp_size
    assert cfg.moe_tp_size * cfg.moe_ep_size == cfg.tp_size, \
        "moe_tp_size * moe_ep_size must equal tp_size"

    os.makedirs(cfg.output_dir, exist_ok=True)
    assert cfg.model_dir, "model_dir is required"

    tik = time.time()
    _convert_and_save_hf(cfg)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - tik))
    print(f"Total time of converting checkpoints: {elapsed}")

    return cfg.output_dir


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen2/InternVL LLM checkpoint to TRT-LLM format.",
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="tllm_checkpoint")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use_weight_only", action="store_true", default=False)
    parser.add_argument("--weight_only_precision", const="int8", type=str,
                        nargs="?", default="int8",
                        choices=["int8", "int4", "int4_gptq"])
    parser.add_argument("--disable_weight_only_quant_plugin",
                        action="store_true", default=False)
    parser.add_argument("--smoothquant", "-sq", type=float, default=None)
    parser.add_argument("--per_channel", action="store_true", default=False)
    parser.add_argument("--per_token", action="store_true", default=False)
    parser.add_argument("--int8_kv_cache", action="store_true", default=False)
    parser.add_argument("--per_group", action="store_true", default=False)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--calib_dataset", type=str,
                        default="ccdv/cnn_dailymail")
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument("--use_parallel_embedding", action="store_true",
                        default=False)
    parser.add_argument("--embedding_sharding_dim", type=int, default=0,
                        choices=[0, 1])
    parser.add_argument("--use_embedding_sharing", action="store_true",
                        default=False)
    parser.add_argument("--moe_tp_size", type=int, default=-1)
    parser.add_argument("--moe_ep_size", type=int, default=-1)
    parser.add_argument("--disable_rope_scaling", action="store_true",
                        default=False)
    return parser.parse_args()


def main():
    args = _parse_arguments()
    convert_llm(**vars(args))


if __name__ == "__main__":
    main()
