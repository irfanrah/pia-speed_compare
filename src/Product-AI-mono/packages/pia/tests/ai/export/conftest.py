import os

import pytest
from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR, TEST_NAME_SPACE
from pia.utils.api.hugging_face import HFModelDownloader


@pytest.fixture(scope="module", autouse=True)
def model_path() -> str:    
    save_dir = ASSETS_MODEL_SAVE_DIR
    save_file_name = "YOLOV8S.pt"
    model_file_path = os.path.join(save_dir, save_file_name)
    downloader = HFModelDownloader(namespace=TEST_NAME_SPACE)
    downloader.download(
        repo_id="YOLOV8S",
        save_dir=save_dir,
        file_name=save_file_name,
    )
    return model_file_path
