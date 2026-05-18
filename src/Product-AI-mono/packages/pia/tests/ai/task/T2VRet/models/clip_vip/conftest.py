import os

import pytest
from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader

CLIPVIP_TORCH_HF_REPO_NAME = "T2V_CLIPVIP_MSRVTT"
CLIPVIP_TORCH_HF_FILE_NAME = f"{CLIPVIP_TORCH_HF_REPO_NAME}.pt"


@pytest.fixture(scope="module", autouse=True)
def download_clip_vip_model():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=CLIPVIP_TORCH_HF_REPO_NAME,
        file_name=CLIPVIP_TORCH_HF_FILE_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
    )


@pytest.fixture
def model_path():
    """Fixture to provide the model path for testing."""
    return os.path.join(ASSETS_MODEL_SAVE_DIR, CLIPVIP_TORCH_HF_FILE_NAME)
