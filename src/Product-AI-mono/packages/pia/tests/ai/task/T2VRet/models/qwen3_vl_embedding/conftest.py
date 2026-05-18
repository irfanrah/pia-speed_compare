from typing import Generator, List, Tuple

import numpy as np
import pytest
import torch
from pia.ai.device import load_model_backend
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader

QWEN3VL_REPO_NAME     = "Qwen3-VL-Embedding-2B"
QWEN3VL_FP8_REPO_NAME = "Qwen3-VL-Embedding-2B-FP8"


def setup_dummy_video(sequence_length: int = 8) -> np.ndarray:
    """Returns a single [S, H, W, C] uint8 video — not a fixture so tests can call it directly."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=(sequence_length, 360, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def setup_dummy_texts() -> List[str]:
    return [
        "A person in white forcibly pulls a person in yellow, who is on the ground.",
        "Someone in black roughly grabs a seated person in white.",
        "A man in red, appearing drunk, is being held up by someone in beige pants.",
    ]


@pytest.fixture(scope="module", autouse=False)
def download_qwen3vl_model():
    hf_downloader = HFModelDownloader(namespace="Qwen")
    hf_downloader.download(
        repo_id=QWEN3VL_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        snapshot=True,
    )


@pytest.fixture(scope="module", autouse=False)
def download_qwen3vl_fp8_model():
    hf_downloader = HFModelDownloader(namespace="PIA-SPACE-LAB")
    hf_downloader.download(
        repo_id=QWEN3VL_FP8_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        snapshot=True,
    )


@pytest.fixture(scope="module")
def setup_qwen3vl_model() -> Generator[Tuple[PiaTorchModel, T2VRetConfig], None, None]:
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(
        model_path=f"Qwen/{QWEN3VL_REPO_NAME}",
        device=device,
        tile_config=None,
        model_name=QWEN3VL_REPO_NAME,
        temporal_size=8,
        img_size=[None, None],
    )
    model = PiaTorchModel(
        target_task="RET",
        target_model="Qwen3VLEmbedding",
        config=config,
    )
    yield model, config

    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def setup_qwen3vl_fp8_model() -> Generator[Tuple[PiaTorchModel, T2VRetConfig], None, None]:
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(
        model_path=f"PIA-SPACE-LAB/{QWEN3VL_FP8_REPO_NAME}",
        device=device,
        tile_config=None,
        model_name=QWEN3VL_FP8_REPO_NAME,
        temporal_size=8,
        img_size=[None, None],
    )
    model = PiaTorchModel(
        target_task="RET",
        target_model="Qwen3VLEmbedding",
        config=config,
    )
    yield model, config

    del model
    torch.cuda.empty_cache()