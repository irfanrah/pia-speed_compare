import os
from typing import List, Tuple

import numpy as np
import pytest
import torch
from pia.ai.device import load_model_backend
from pia.ai.tasks.T2VRet.base import T2VRetConfig, T2VRetONNXConfig
from pia.ai.tasks.T2VRet.models.clip4clip.main import Clip4Clip
from pia.ai.tasks.T2VRet.models.clip4clip.model import CLIP4Clip
from pia.tests.ai.task.T2VRet.models.clip4clip.conftest import (
    ASSETS_MODEL_SAVE_DIR,
    C4C_TORCH_MODEL_FILE,
    setup_dummy_video,
    setup_torch_model,
)


@pytest.fixture
def setup_torch_half_model() -> Tuple[Clip4Clip, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    model, config = setup_torch_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, C4C_TORCH_MODEL_FILE),
        use_half_precision=True
    )
    return model, config


@pytest.fixture
def torch_setup_tiled_model() -> Tuple[Clip4Clip, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    model, config = setup_torch_model(
        device=device,
        model_path=os.path.join(ASSETS_MODEL_SAVE_DIR, C4C_TORCH_MODEL_FILE),
        tile_config="S",
    )
    return model, config


def test_clip4clip_torch_model_video_vector(
    setup_torch_default_model: Tuple[Clip4Clip, T2VRetConfig],
):
    """Encodes frame sequence.

    Args:
        torch_model (Clip4Clip): CLIP4Clip model instance.
        dummy_video (np.ndarray): Dummy input video array. Shape is (Batch size,
            Sequence length, height, width, channel).
    """
    torch_default_model, torch_default_model_config = setup_torch_default_model
    dummy_video = setup_dummy_video(sequence_length=torch_default_model_config.temporal_size)

    vis_vector = torch_default_model(video=dummy_video)

    batch_size, sequence_length, height, width, channel = dummy_video.shape
    num_tiles = 1
    embedding_dimension = 512

    assert vis_vector.shape == (batch_size, num_tiles, embedding_dimension)
    assert isinstance(vis_vector, torch.Tensor)


def test_clip4clip_torch_model_video_vector_tiled(
    torch_setup_tiled_model: Tuple[Clip4Clip, T2VRetConfig],
):
    """Encodes frame sequence with tiled frames."""
    torch_model_tiled, torch_model_tiled_config = torch_setup_tiled_model
    dummy_video = setup_dummy_video(sequence_length=torch_model_tiled_config.temporal_size)

    vis_vector = torch_model_tiled(video=dummy_video)

    batch_size, sequence_length, height, width, channel = dummy_video.shape
    num_tiles = 19
    embedding_dimension = 512

    assert vis_vector.shape == (batch_size, num_tiles, embedding_dimension)
    assert isinstance(vis_vector, torch.Tensor)


def test_clip4clip_torch_model_text_vector(
    setup_torch_default_model: Tuple[Clip4Clip, T2VRetConfig],
    setup_dummy_texts: List[str]
):
    torch_model, torch_model_config = setup_torch_default_model
    dummy_texts = setup_dummy_texts

    txt_vector = torch_model(text=dummy_texts[0])

    num_captions = len(dummy_texts)
    embedding_dimension = 512

    assert txt_vector.shape == (1, embedding_dimension)
    assert isinstance(txt_vector, torch.Tensor)

    txt_vector = torch_model(text=dummy_texts)

    assert txt_vector.shape == (num_captions, embedding_dimension)
    assert isinstance(txt_vector, torch.Tensor)


def test_clip4clip_torch_model_similatiry(
        setup_torch_default_model: Tuple[Clip4Clip, T2VRetConfig],
        setup_dummy_texts: List[str]):
    torch_model, torch_model_config = setup_torch_default_model
    dummy_video = setup_dummy_video(sequence_length=torch_model_config.temporal_size)
    dummy_texts = setup_dummy_texts

    similarity = torch_model(video=dummy_video, text=dummy_texts)

    num_captions = len(dummy_texts)
    num_video_batches = dummy_video.shape[0]

    assert similarity.shape == (num_captions, num_video_batches)
    assert isinstance(similarity, np.ndarray)
    assert np.all((similarity >= 0) & (similarity <= 1))


def test_clip4clip_torch_model_similarity_tiled(
    torch_setup_tiled_model: Tuple[Clip4Clip, T2VRetConfig],
    setup_torch_default_model: Tuple[Clip4Clip, T2VRetConfig],
    setup_dummy_texts: List[str]
):
    # inference tiled
    torch_model_tiled, torch_model_config_tiled = torch_setup_tiled_model

    dummy_video = setup_dummy_video(sequence_length=torch_model_config_tiled.temporal_size)
    dummy_texts = setup_dummy_texts

    similarity_tiled = torch_model_tiled(video=dummy_video, text=dummy_texts)

    # inference no tile
    torch_model, torch_model_config = setup_torch_default_model
    similarity = torch_model(video=dummy_video, text=dummy_texts)
    assert similarity.shape == (len(dummy_texts), len(dummy_video))

    num_captions = len(dummy_texts)
    num_video_batches = dummy_video.shape[0]

    assert similarity_tiled.shape == (num_captions, num_video_batches)
    assert isinstance(similarity_tiled, np.ndarray)
    assert np.all((similarity_tiled >= 0) & (similarity_tiled <= 1))


def test_clip4clip_model2pt(setup_torch_default_model):
    torch_model, _ = setup_torch_default_model

    output_pt_path = os.path.join(ASSETS_MODEL_SAVE_DIR, "test_model_clip4clip.pt")
    torch_model.model2pt(output_pt_path)

    model = torch.load(output_pt_path, weights_only=False)

    assert isinstance(model, CLIP4Clip)


def test_clip4clip_pt2onnx(setup_torch_default_model):
    torch_model, _ = setup_torch_default_model

    # visual
    onnx_config = T2VRetONNXConfig(output_dir=ASSETS_MODEL_SAVE_DIR, split_part="visual")  #
    torch_model.export(onnx_config)
    # textual
    onnx_config = T2VRetONNXConfig(output_dir=ASSETS_MODEL_SAVE_DIR, split_part="textual")  #
    torch_model.export(onnx_config)
    # embedding
    onnx_config = T2VRetONNXConfig(output_dir=ASSETS_MODEL_SAVE_DIR, split_part="embedding")  #
    torch_model.export(onnx_config)
