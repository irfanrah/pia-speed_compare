"""InternVL3 LLM 체크포인트 변환 테스트.

HuggingFace 모델 디렉터리를 TRT-LLM 체크포인트 형식으로 변환하고
필수 파일이 생성되는지 확인한다. CUDA 환경에서만 실행된다.
"""

import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_convert_llm(model_dir, tllm_checkpoint_dir):
    """TRT-LLM 체크포인트 디렉터리와 파일 생성 여부를 검증한다."""
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
