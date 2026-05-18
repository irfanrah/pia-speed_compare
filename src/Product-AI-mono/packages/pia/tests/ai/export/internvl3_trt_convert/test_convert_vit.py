"""InternVL3 VIT 엔진 빌드 테스트.

HF 모델을 기반으로 VIT ONNX를 생성하고 TensorRT 엔진(.trt)까지
빌드되는지 확인한다. CUDA 환경에서만 실행된다.
"""

import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_vit_engine(model_dir, vit_onnx_file, vit_trt_file, test_frame_path):
    """VIT ONNX 및 TRT 엔진 파일이 생성되는지 검증한다."""
    from pia.ai.exports.internvl3_trt_convert import build_vit_engine

    result = build_vit_engine(
        pretrained_model_path=model_dir,
        onnx_file=vit_onnx_file,
        trt_file=vit_trt_file,
        image_path=test_frame_path,
        dtype="bfloat16",
        min_bs=1,
        opt_bs=13,
        max_bs=32,
    )

    assert result == vit_trt_file
    assert os.path.isfile(vit_onnx_file)
    assert os.path.isfile(vit_trt_file)
