import cv2
import numpy as np
import torch
from pia.vision.preprocessing import (
    batch_color_filter,
    batch_convert_colors,
    color_filtering,
    convert_colors,
    cv_bgr2rgb_batch,
    numba_bgr2rgb_batch_cf,
    numba_bgr2rgb_batch_cl,
    torch_bgr2rgb_batch_cf,
    torch_bgr2rgb_batch_cl,
)


# ───────────────────────── Numba 테스트 ─────────────────────────
def test_numba_bgr2rgb():
    # channel-first (B, 3, H, W)
    bgr_cf = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
    rgb_cf = numba_bgr2rgb_batch_cf(bgr_cf)
    assert np.array_equal(rgb_cf, bgr_cf[:, [2, 1, 0], :, :])  # 전체 비교

    # channel-last (B, H, W, 3)
    bgr_cl = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    rgb_cl = numba_bgr2rgb_batch_cl(bgr_cl)
    assert np.array_equal(rgb_cl, bgr_cl[..., ::-1])  # 전체 비교
    # OpenCV 결과와도 동일해야 함
    assert np.array_equal(rgb_cl[0], cv2.cvtColor(bgr_cl[0], cv2.COLOR_BGR2RGB))


# ───────────────────────── OpenCV 테스트 ─────────────────────────
def test_cv_bgr2rgb():
    bgr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    bgr_orig = bgr.copy()
    cv_bgr2rgb_batch([bgr])  # in-place 변환
    assert np.array_equal(bgr, bgr_orig[..., ::-1])  # 전체 비교


# ───────────────────────── PyTorch 테스트 ─────────────────────────
def test_torch_bgr2rgb():
    # channel-first (B, 3, H, W)
    bgr_cf = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
    rgb_cf = torch_bgr2rgb_batch_cf(bgr_cf)
    assert torch.equal(rgb_cf, bgr_cf[:, [2, 1, 0], :, :])  # 전체 비교

    # channel-last (B, H, W, 3)
    bgr_cl = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
    rgb_cl = torch_bgr2rgb_batch_cl(bgr_cl)
    assert torch.equal(rgb_cl, bgr_cl[..., [2, 1, 0]])  # 전체 비교

    # OpenCV 결과와 비교 (numpy↔torch 변환)
    cv_rgb = torch.from_numpy(cv2.cvtColor(bgr_cl[0].numpy(), cv2.COLOR_BGR2RGB))
    assert torch.equal(rgb_cl[0], cv_rgb)


# ───────────────────────── color 테스트 ─────────────────────────
def test_convert_colors_single():
    img = np.ones((5, 5, 3), dtype=np.uint8) * 127
    conditions = ["rgb", "gray", "bgr"]

    result = convert_colors(img, conditions)

    # 변환된 키들이 포함되어야 함
    assert "rgb" in result
    assert "gray" in result
    assert "bgr" not in result

    # gray는 단일 채널이어야 함
    assert len(result["gray"].shape) == 2


def test_convert_colors_batch_multiple():
    batch = [
        np.ones((5, 5, 3), dtype=np.uint8) * 50,
        np.ones((5, 5, 3), dtype=np.uint8) * 200,
    ]
    conditions = ["rgb", "hsv"]

    result = batch_convert_colors(batch, conditions)

    # 두 개 이미지 넣었으니 각 key에 리스트 길이가 2
    assert len(result["rgb"]) == 2
    assert len(result["hsv"]) == 2
    assert isinstance(result["rgb"][0], np.ndarray)


def test_ignore_invalid_condition():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    result = convert_colors(img, ["invalid", "rgb"])

    # invalid는 무시되어야 함
    assert "invalid" not in result
    assert "rgb" in result


# ───────────────────────── color filtering 테스트 ─────────────────────────
def test_color_filtering_basic():
    img = np.ones((20, 20, 3), dtype=np.uint8) * 200
    lower = (100, 100, 100)
    upper = (255, 255, 255)

    mask = color_filtering(img, lower, upper, kernel_size=(3, 3))

    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    # 모든 픽셀이 white 영역이어야 함
    assert np.all(mask == 255)


def test_color_filtering_operations():
    img = np.ones((20, 20, 3), dtype=np.uint8) * 200
    lower = (100, 100, 100)
    upper = (255, 255, 255)

    mask = color_filtering(
        img, lower, upper, kernel_size=(3, 3), operations=[("erode", 1)]
    )
    assert mask.shape == (20, 20)


def test_batch_color_filter():
    batch = [
        np.ones((10, 10, 3), dtype=np.uint8) * 50,
        np.ones((10, 10, 3), dtype=np.uint8) * 200,
    ]
    range = [
        [np.array((0, 0, 0)), np.array((60, 60, 60))],
        [np.array((0, 0, 0)), np.array((255, 10, 10))]
    ]

    result = batch_color_filter(
        batch,
        range=range,
        kernel_size=(3, 3),
        operations=[("close", 1)],
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(mask, np.ndarray) for mask in result)
