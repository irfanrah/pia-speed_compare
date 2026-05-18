import gc
import os
import shutil

import pytest
from pia.tests.test_config import ASSETS_IMAGE_SAVE_DIR, ASSETS_MODEL_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader
from pia.utils.api.nas import NasConfig, NasManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
"""Pytest fixtures for InternVL3 TRT-LLM export/conversion tests.

이 모듈은 모델 다운로드, 테스트 입력 프레임 다운로드, 엔진/체크포인트 경로
준비 및 테스트 종료 후 임시 산출물 정리를 담당한다.
"""

HF_REPO_ID = "OpenGVLab/InternVL3-2B"

TEST_FRAME_FILE_NAME = "frame_0.jpg"
TEST_FRAME_SAVE_DIR = os.path.join("assets", "images")

# output dirs (test-only, cleaned up after)
TLLM_CHECKPOINT_TEST_DIR = os.path.join(
    ASSETS_MODEL_SAVE_DIR, HF_REPO_ID, "tllm_checkpoint_test"
)
TRT_ENGINE_TEST_DIR = os.path.join(
    ASSETS_MODEL_SAVE_DIR, HF_REPO_ID, "trt_engines_test"
)
VIT_ONNX_TEST_FILE = os.path.join(
    ASSETS_MODEL_SAVE_DIR, HF_REPO_ID, "vision_encoder_bfp16_test.onnx"
)
VIT_TRT_TEST_FILE = os.path.join(
    ASSETS_MODEL_SAVE_DIR, HF_REPO_ID, "vision_encoder_bfp16_test.trt"
)


# ---------------------------------------------------------------------------
# Downloads (session scope - run once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def download_model():
    """HuggingFace에서 InternVL3-2B 모델을 세션당 1회 다운로드한다."""
    downloader = HFModelDownloader()
    downloader.download(
        repo_id=HF_REPO_ID, save_dir=ASSETS_MODEL_SAVE_DIR, snapshot=True
    )


@pytest.fixture(scope="session", autouse=True)
def download_test_frame():
    """NAS에서 테스트 프레임 이미지를 내려받아 엔진 빌드 입력으로 사용한다."""
    config = NasConfig()
    nas_manager = NasManager(config)
    save_path = os.path.join(TEST_FRAME_SAVE_DIR, TEST_FRAME_FILE_NAME)
    nas_url = nas_manager.get_nas_path(save_path)
    downloaded = nas_manager.download_file(nas_url, save_path)
    assert os.path.exists(downloaded), f"Frame download failed: {save_path}"


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def model_dir():
    """다운로드된 HF 모델 디렉터리 경로를 제공한다."""
    return os.path.join(ASSETS_MODEL_SAVE_DIR, HF_REPO_ID)


@pytest.fixture(scope="session")
def test_frame_path():
    """VIT 엔진 빌드용 테스트 프레임 이미지 경로를 제공한다."""
    return os.path.join(TEST_FRAME_SAVE_DIR, TEST_FRAME_FILE_NAME)


@pytest.fixture(scope="session")
def tllm_checkpoint_dir():
    """TRT-LLM 체크포인트 테스트 출력 디렉터리를 제공한다."""
    return TLLM_CHECKPOINT_TEST_DIR


@pytest.fixture(scope="session")
def trt_engine_dir():
    """TRT 엔진 테스트 출력 디렉터리를 제공한다."""
    return TRT_ENGINE_TEST_DIR


@pytest.fixture(scope="session")
def vit_onnx_file():
    """VIT ONNX 테스트 출력 파일 경로를 제공한다."""
    return VIT_ONNX_TEST_FILE


@pytest.fixture(scope="session")
def vit_trt_file():
    """VIT TRT 엔진 테스트 출력 파일 경로를 제공한다."""
    return VIT_TRT_TEST_FILE


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def cleanup_gpu():
    """테스트 단위로 GPU 메모리를 정리한다."""
    yield
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_outputs():
    """세션 종료 시 테스트 산출물(엔진/체크포인트/ONNX)을 정리한다."""
    yield
    for d in (TLLM_CHECKPOINT_TEST_DIR, TRT_ENGINE_TEST_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
    for f in (
        VIT_ONNX_TEST_FILE,
        VIT_ONNX_TEST_FILE + ".data",
        VIT_TRT_TEST_FILE,
    ):
        if os.path.isfile(f):
            os.remove(f)
