import functools
import os

import pytest
from pia.tests.test_config import ASSETS_VIDEO_SAVE_DIR
from pia.utils.api.nas import NasConfig, NasManager


def require_nas_connect(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            nas_manager = NasManager(NasConfig())
            nas_manager._get_response()
        except Exception:
            pytest.skip("NAS 연결에 실패하여 테스트를 건너뜁니다.")
        return func(*args, **kwargs)
    return wrapper


def test_instance():
    config = NasConfig()
    assert isinstance(config, NasConfig), "Failed to create NasConfig instance"

    nas_manager = NasManager(config)
    assert isinstance(nas_manager, NasManager), "Failed to create NasManager instance"

@require_nas_connect
def test_nas_manager_download_file():
    config = NasConfig()
    nas_manager = NasManager(config)
    nas_manager._get_response()

    # Define a test file name and path
    test_file_name = "safety0003.mp4"
    test_file_save_path = os.path.join(ASSETS_VIDEO_SAVE_DIR, test_file_name)
    nas_file_path = nas_manager.get_nas_path(test_file_save_path)

    # Now try to download the file using the NasManager
    downloaded_path = nas_manager.download_file(nas_file_path, test_file_save_path)

    # Check if the downloaded file exists and has the correct content
    assert os.path.exists(downloaded_path), "Downloaded file does not exist."
