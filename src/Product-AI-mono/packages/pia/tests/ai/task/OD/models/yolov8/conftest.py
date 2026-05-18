import os

import cv2
import pytest
from pia.tests.test_config import ASSETS_IMAGE_SAVE_DIR, ASSETS_MODEL_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader

PEOPLE_SAMPLE_IMAGE_PATH = os.path.join(ASSETS_IMAGE_SAVE_DIR, "people.jpeg")


@pytest.fixture
def img():
    if not os.path.exists(PEOPLE_SAMPLE_IMAGE_PATH):
        raise FileNotFoundError(f"Test image {PEOPLE_SAMPLE_IMAGE_PATH} does not exist.")
    img = cv2.imread(PEOPLE_SAMPLE_IMAGE_PATH)
    if img is None:
        raise ValueError(f"Failed to read image from {PEOPLE_SAMPLE_IMAGE_PATH}.")
    return img


@pytest.fixture
def model_path():
    save_file_name = "YOLOV8S.pt"
    save_path = os.path.join(ASSETS_MODEL_SAVE_DIR, save_file_name)
    downloader = HFModelDownloader()
    downloader.download(
        repo_id="YOLOV8S",
        save_dir=ASSETS_MODEL_SAVE_DIR,
        file_name=save_file_name
    )
    return save_path
