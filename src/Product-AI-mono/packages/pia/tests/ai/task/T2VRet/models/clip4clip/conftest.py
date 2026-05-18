import os
from typing import List, Literal, Tuple

import cv2
import numpy as np
import pytest
from pia.ai.device import load_model_backend
from pia.ai.model import PiaONNXTensorRTModel, PiaTorchModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.ai.tasks.T2VRet.models.clip4clip.main import Clip4Clip
from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR, ASSETS_VIDEO_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader
from pia.vision.load.opencv import get_sequence

C4C_HF_REPO_NAME = "T2V_CLIP4CLIP_MSRVTT"
C4C_TORCH_MODEL_FILE = C4C_HF_REPO_NAME + ".pt"
C4C_ONNX_HF_REPO_NAME = "CLIP4CLIP_ONNX"
ONNX_VIS_MODEL_FILE = "visual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx"
ONNX_TXT_MODEL_FILE = "textual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx"
ONNX_TOKEN_EMBEDDER_FILE = "embedding_f32_op14_clip4clip_msrvtt_b128_ep5.onnx"


@pytest.fixture
def setup_dummy_texts() -> List[str]:
    texts = [
        "A person in white forcibly pulls a person in yellow, who is on the ground.",
        "Someone in black roughly grabs a seated person in white.",
        "A man in red, appearing drunk, is being held up by someone in beige pants.",
    ]
    return texts


@pytest.fixture
def video_encoder():
    device = load_model_backend("cuda", type="str")
    model = setup_onnx_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, ONNX_VIS_MODEL_FILE),
    )
    return model


@pytest.fixture
def text_encoder():
    device = load_model_backend("cuda", type="str")
    model = setup_onnx_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, ONNX_TXT_MODEL_FILE),
    )
    return model


@pytest.fixture
def token_embedder():
    device = load_model_backend("cuda", type="str")
    model = setup_onnx_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, ONNX_TOKEN_EMBEDDER_FILE),
    )
    return model


@pytest.fixture
def torch_model():
    device = load_model_backend("cuda", type="str")
    model, _ = setup_torch_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, C4C_TORCH_MODEL_FILE),
    )
    return model


@pytest.fixture(scope="module", autouse=True)
def download_models():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=C4C_HF_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        file_name=C4C_TORCH_MODEL_FILE
    )


@pytest.fixture(scope="module", autouse=True)
def download_onnx_model():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=C4C_ONNX_HF_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        file_name=ONNX_VIS_MODEL_FILE
    )
    hf_downloader.download(
        repo_id=C4C_ONNX_HF_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        file_name=ONNX_TXT_MODEL_FILE
    )
    hf_downloader.download(
        repo_id=C4C_ONNX_HF_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        file_name=ONNX_TOKEN_EMBEDDER_FILE
    )


def setup_dummy_video(sequence_length: int) -> np.ndarray:
    video_path = os.path.join(ASSETS_VIDEO_SAVE_DIR, "safety0001.mp4")
    cap = cv2.VideoCapture(video_path)
    batch_size = 3
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    channel = 3

    batch = []
    start_frame_ids = (2, 4, 8)
    for batch_id in range(batch_size):
        sequence = get_sequence(
            cap=cap,
            start_frame_id=start_frame_ids[batch_id],
            sequence_length=sequence_length,
        )
        batch.append(sequence)

    video = np.asarray(batch)
    assert video.shape == (batch_size, sequence_length, height, width, channel)

    return video


@pytest.fixture
def setup_sample_videos():
    """Download Sample Videos
    Return:
        list: List of paths for sample video files

    """
    videos = [
        "safety0001.mp4",
        "safety0002.mp4",
        "safety0003.mp4",
    ]
    return [os.path.join(ASSETS_VIDEO_SAVE_DIR, video) for video in videos]


@pytest.fixture
def setup_torch_default_model() -> Tuple[Clip4Clip, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    model, config = setup_torch_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, C4C_TORCH_MODEL_FILE),
    )
    return model, config


def setup_torch_model(
    device: Literal["cuda", "cpu", "mps"],
    model_path: str,
    tile_config: str = None,
    use_half_precision: bool = False
) -> Tuple[PiaTorchModel, T2VRetConfig]:

    config = T2VRetConfig(
        model_path=model_path,
        device=device,
        tile_config=tile_config,
        use_half_precision=use_half_precision,
    )
    model = PiaTorchModel(
        target_task=1,
        target_model=0,
        config=config,  # T2VRet  # clip4clip
    )

    return model, config


def setup_onnx_model(
    device: Literal["cuda", "cpu", "mps"],
    model_path: str,
) -> Tuple[PiaTorchModel, T2VRetConfig]:

    return PiaONNXTensorRTModel(model_path=model_path, device=device)
