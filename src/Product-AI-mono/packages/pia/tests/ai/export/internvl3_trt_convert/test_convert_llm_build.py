"""InternVL3 LLM 엔진 빌드 테스트.

HF 모델을 TRT-LLM 체크포인트로 변환한 뒤,
TensorRT 엔진 디렉터리가 생성되는지 확인한다.
CUDA 환경에서만 실행된다.
"""

import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_llm_engine(model_dir,tllm_checkpoint_dir, trt_engine_dir):
    """체크포인트 변환과 엔진 빌드 결과 파일 생성을 검증한다."""
    from pia.ai.exports.internvl3_trt_convert import build_llm_engine

    from pia.ai.exports.internvl3_trt_convert import convert_llm

    result = convert_llm(
        model_dir=model_dir,
        output_dir=tllm_checkpoint_dir,
        dtype="bfloat16",
        load_model_on_cpu=True,
        disable_rope_scaling=True,
    )

    assert result == tllm_checkpoint_dir
    assert os.path.isdir(tllm_checkpoint_dir)
    assert os.path.isfile(os.path.join(tllm_checkpoint_dir, "config.json"))
    assert os.path.isfile(os.path.join(tllm_checkpoint_dir, "rank0.safetensors"))

    result = build_llm_engine(
        checkpoint_dir=tllm_checkpoint_dir,
        output_dir=trt_engine_dir,
        max_batch_size=4,
        max_input_len=3584,
        max_seq_len=3840,
        max_num_tokens=15200,
        gemm_plugin="bfloat16",
        max_multimodal_len=14000,
    )

    assert result == trt_engine_dir
    assert os.path.isdir(trt_engine_dir)
    assert os.path.isfile(os.path.join(trt_engine_dir, "config.json"))
