import numpy as np
import pytest
import torch
from pia.utils.exception.model_handler import (
    validate_text_embedding_vector,
    validate_tile_config,
    validate_video_embedding_vector,
    validate_video_shape,
)


def test_validate_tile_config_ok():
    validate_tile_config({"size": 256, "overlap_ratio": 0.5})


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {"size": 256},  # 키 누락
        {"size": 256, "overlap_ratio": 0.5, "extra": 1},  # 키 추가
        {"size": "256", "overlap_ratio": 0.5},  # 타입 불일치
        {"size": 256, "overlap_ratio": 1},  # float 아님(int)
        {"size": 256, "overlap_ratio": -0.1},  # 범위 밖
        {"size": 256, "overlap_ratio": 1.1},  # 범위 밖
    ],
)
def test_validate_tile_config_raises(bad_cfg):
    with pytest.raises(ValueError):
        validate_tile_config(bad_cfg)


def test_validate_video_shape_ok_4d():
    video = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    validate_video_shape(video)


def test_validate_video_shape_ok_5d():
    video = np.zeros((2, 8, 64, 64, 3), dtype=np.uint8)
    validate_video_shape(video)


def test_validate_video_shape_raises():
    video = np.zeros((8, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_video_shape(video)


def test_validate_text_embedding_vector_ok():
    t = torch.zeros((5, 1024))
    validate_text_embedding_vector(t)


@pytest.mark.parametrize("shape", [(1024,), (2, 5, 1024)])
def test_validate_text_embedding_vector_raises(shape):
    t = torch.zeros(shape)
    with pytest.raises(ValueError):
        validate_text_embedding_vector(t)


def test_validate_video_embedding_vector_ok():
    v = torch.zeros((4, 16, 1024))
    validate_video_embedding_vector(v)


@pytest.mark.parametrize("shape", [(16, 1024), (1, 4, 16, 1024)])
def test_validate_video_embedding_vector_raises(shape):
    v = torch.zeros(shape)
    with pytest.raises(ValueError):
        validate_video_embedding_vector(v)
