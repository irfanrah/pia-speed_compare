import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from pia.ai.exports import trt, yolo
from pia.ai.model import PiaONNXTensorRTModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not shutil.which("trtexec"), reason="trtexec is not installed")
def test_onnx2trt_invokes_trtexec(model_path: str):
    assert os.path.exists(model_path), f"Model file {model_path} does not exist."
    model_file_name = os.path.splitext(model_path)[0]
    yolo.yolo2onnx(model_path, opset=17, overwrite=True)
    engine = model_file_name + ".engine"
    onnx = model_file_name + ".onnx"
    try:
        trt.onnx2trt(
            onnx_path=onnx,
            engine_path=engine,
            device=0,
            fp16=True,
            overwrite=True,
        )
    except subprocess.CalledProcessError as e:
        # 실행 실패 시 로깅 (예: trtexec이 없을 경우 등)
        print("trtexec execution failed")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        assert False, "trtexec execution failed"

    # 실행 후 파일이 생성되었는지 확인
    assert os.path.exists(engine), "TRT engine file was not created"
    model = PiaONNXTensorRTModel(engine, device="cuda")
    assert model is not None, "Can't load engine file"



def test_yolo2onnx_calls_ultralytics(model_path):
    yolo.yolo2onnx(model_path, opset=17, overwrite=True, half=False)
    assert os.path.exists(Path(model_path).with_suffix(".onnx")), "onnx file was not created"
    yolo.yolo2onnx(model_path, opset=17, overwrite=True, half=True)
    assert os.path.exists(Path(model_path).with_suffix(".onnx")), "onnx file was not created"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not shutil.which("trtexec"), reason="trtexec is not installed")
def test_yolo2trt_delegates_to_onnx(model_path):
    yolo.yolo2trt(model_path, device=0, half=True, overwrite=True)
    assert os.path.exists(Path(model_path).with_suffix(".engine")), "TensorRT engine file was not created"
