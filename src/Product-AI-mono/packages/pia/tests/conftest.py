import os

import pytest
from pia.tests.test_config import (
    ASSETS_IMAGE_SAVE_DIR,
    ASSETS_VIDEO_SAVE_DIR,
    TEST_IMAGES,
    TEST_VIDEOS,
)
from pia.utils.api.nas import NasConfig, NasManager


@pytest.fixture(scope="session", autouse=True)
def download_nas_images():
    config = NasConfig()
    nas_manager = NasManager(config)
    for file_name in TEST_IMAGES:
        test_file_save_path = os.path.join(ASSETS_IMAGE_SAVE_DIR, file_name)
        nas_file_path = nas_manager.get_nas_path(test_file_save_path)
        downloaded_path = nas_manager.download_file(nas_file_path, test_file_save_path)
        assert os.path.exists(downloaded_path), f"Downloaded file {file_name} does not exist."


@pytest.fixture(scope="session", autouse=True)
def download_nas_videos():
    config = NasConfig()
    nas_manager = NasManager(config)
    for file_name in TEST_VIDEOS:
        test_file_save_path = os.path.join(ASSETS_VIDEO_SAVE_DIR, file_name)
        nas_file_path = nas_manager.get_nas_path(test_file_save_path)
        downloaded_path = nas_manager.download_file(nas_file_path, test_file_save_path)
        assert os.path.exists(downloaded_path), f"Downloaded file {file_name} does not exist."
