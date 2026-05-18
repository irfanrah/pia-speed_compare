import numpy as np
import pytest
import torch
from pia.vision.preprocessing.t2v import PE_video_preprocess, video_preprocess
from torchvision import transforms

B, S, H, W, C = 2, 3, 1080, 1920, 3
IMG_SIZE = (224, 224)
TO_TENSOR = transforms.Compose([transforms.ToTensor()])
RESIZE_TO_IMG = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((IMG_SIZE[1], IMG_SIZE[0]))])


def _rand_video(shape):
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)


def test_video_preprocess_no_batch_no_tile_cpu():
    # (S, H, W, C) 입력 → 배치 차원 자동 추가
    video = _rand_video((S, H, W, C))
    vt, mask = video_preprocess(
        video,
        device="cpu",
        tile_size=None,
        img_size=IMG_SIZE,
        transform=TO_TENSOR,
    )
    assert isinstance(vt, torch.Tensor) and isinstance(mask, torch.Tensor)
    assert vt.shape == (1, 1, S, C, IMG_SIZE[1], IMG_SIZE[0])
    assert mask.shape == (1, 1, S)
    assert vt.min() >= 0.0 and vt.max() <= 1.0  # ToTensor()로 정규화


def test_video_preprocess_batched_no_tile_cpu():
    video = _rand_video((B, S, H, W, C))
    vt, mask = video_preprocess(
        video,
        device="cpu",
        tile_size=None,
        img_size=IMG_SIZE,
        transform=TO_TENSOR,
    )
    assert vt.shape == (B, 1, S, C, IMG_SIZE[1], IMG_SIZE[0])
    assert mask.shape == (B, 1, S)


def test_video_preprocess_with_tiles():
    video = _rand_video((B, S, H, W, C))
    model_size = [224, 224]

    vt, mask = video_preprocess(
        video,
        device="cpu",
        tile_size="S",
        img_size=IMG_SIZE,
        transform=RESIZE_TO_IMG,
    )  # tile_s = [224, 224]

    NUM_TILES = 6 * 3 + 1  # 6*3 타일 + 1 = 19

    assert vt.shape == (B, NUM_TILES, S, C, model_size[1], model_size[0])
    assert mask.shape == (B, NUM_TILES, S)

    vt, mask = video_preprocess(
        video,
        device="cpu",
        tile_size="M",
        img_size=IMG_SIZE,
        transform=RESIZE_TO_IMG,
    )  # tile_m = [336, 336]

    NUM_TILES = 3 * 5 + 1  # 3*5 타일 + 1 = 16

    assert vt.shape == (B, NUM_TILES, S, C, model_size[1], model_size[0])
    assert mask.shape == (B, NUM_TILES, S)

    vt, mask = video_preprocess(
        video,
        device="cpu",
        tile_size="L",
        img_size=IMG_SIZE,
        transform=RESIZE_TO_IMG,
    )  # tile_l = [448, 448]

    NUM_TILES = 2 * 4 + 1  # 2*4 타일 + 1 = 9

    assert vt.shape == (B, NUM_TILES, S, C, model_size[1], model_size[0])
    assert mask.shape == (B, NUM_TILES, S)

    vt, mask = video_preprocess(
        video,
        device="cpu",
        tile_size="no_tile",
        img_size=IMG_SIZE,
        transform=RESIZE_TO_IMG,
    )

    NUM_TILES = 1

    assert vt.shape == (B, 1, S, C, IMG_SIZE[1], IMG_SIZE[0])
    assert mask.shape == (B, 1, S)


def test_video_preprocess_raises_on_bad_shape():
    bad = _rand_video((H, W, C))  # (S, H, W, C)가 아님
    with pytest.raises(ValueError):
        video_preprocess(bad, device="cpu", tile_size=None, img_size=IMG_SIZE, transform=TO_TENSOR)


def test_pe_video_preprocess_basic_no_tile_cpu():
    video = _rand_video((B, S, H, W, C))
    vt, mask = PE_video_preprocess(
        video,
        device="cpu",
        tile_size=None,
        transform=transforms.Compose([transforms.Resize((32, 48)), transforms.ToTensor()]),  # PIL 기반
    )
    assert vt.shape == (B * 1, S, 3, 32, 48)
    assert mask.shape == (B, 1, S)
    assert vt.min() >= 0.0 and vt.max() <= 1.0
