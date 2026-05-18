import os

import pytest
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.CP.base import CPONNXConfig
from pia.tests.test_config import ASSETS_ANNOTATION_SAVE_DIR, ASSETS_IMAGE_SAVE_DIR, ASSETS_MODEL_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader
from pia.utils.api.nas import NasConfig, NasManager

REPO_NAME = "CLIP_EBC_nwpu_rmse_onnx"
MODEL_PATH = os.path.join(ASSETS_MODEL_SAVE_DIR, f"{REPO_NAME}.onnx")

TEST_ANNOTATION_FILES = [
    "songkran_annotations100.xml",
]

TEST_IMAGE_FILES = [
    "songkran37.jpeg",
    "songkran46.jpg",
    "songkran47.jpg",
    "songkran182.png",
    "songkran268.png",
    "songkran289.png",
]


@pytest.fixture(scope="session", autouse=True)
def download_clip_ebc_test_files():
    """NAS에서 clip_ebc 테스트에 필요한 이미지 및 어노테이션 파일을 다운로드하는 Fixture"""
    os.environ["CLIP_EBC_IMAGE_ROOT"] = ASSETS_IMAGE_SAVE_DIR
    config = NasConfig()
    nas_manager = NasManager(config)

    for file_name in TEST_IMAGE_FILES:
        save_path = os.path.join(ASSETS_IMAGE_SAVE_DIR, file_name)
        nas_path = nas_manager.get_nas_path(save_path)
        downloaded_path = nas_manager.download_file(nas_path, save_path)
        assert os.path.exists(downloaded_path), f"Downloaded file {file_name} does not exist."

    for file_name in TEST_ANNOTATION_FILES:
        save_path = os.path.join(ASSETS_ANNOTATION_SAVE_DIR, file_name)
        nas_path = nas_manager.get_nas_path(save_path)
        downloaded_path = nas_manager.download_file(nas_path, save_path)
        assert os.path.exists(downloaded_path), f"Downloaded file {file_name} does not exist."


@pytest.fixture(scope="session", autouse=True)
def ensure_model_downloaded():
    """세션 시작 시 ONNX 모델 파일을 한 번만 다운로드한다."""
    if not os.path.exists(MODEL_PATH):
        HFModelDownloader().download(
            repo_id=REPO_NAME,
            save_dir=ASSETS_MODEL_SAVE_DIR,
            file_name=os.path.basename(MODEL_PATH),
        )


@pytest.fixture
def model():
    """테스트를 위한 모델 객체를 생성하는 Fixture"""
    config = CPONNXConfig(model_path=MODEL_PATH)
    return PiaTorchModel(target_task="CP", target_model="clip_ebc_onnx", config=config)


@pytest.fixture
def annotation_paths():
    """테스트를 위한 어노테이션 파일 경로를 반환하는 Fixture"""
    return os.path.join(ASSETS_ANNOTATION_SAVE_DIR, "songkran_annotations100.xml")


@pytest.fixture
def image_paths():
    """테스트를 위한 이미지 경로 목록을 반환하는 Fixture"""
    return [os.path.join(ASSETS_IMAGE_SAVE_DIR, f) for f in TEST_IMAGE_FILES]
