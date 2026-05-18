import numpy as np
import pytest
import torch
from pia.vision.roi.roi_manager import (
    batch_crop_region,
    cv_crop_region,
    torch_crop_region,
)


def make_dummy_polygon():
    # 단순 사각형 ROI (10,10) ~ (20,20)
    return np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32)


def make_numpy_image(h=50, w=50, c=3):
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def make_torch_image(h=50, w=50, c=3, device="cpu"):
    return torch.randint(0, 256, (c, h, w), dtype=torch.uint8, device=device)


def test_cv_crop_region_basic():
    img = make_numpy_image()
    poly = make_dummy_polygon()
    cropped = cv_crop_region(img, poly.tolist())
    assert isinstance(cropped, np.ndarray)
    h, w, _ = cropped.shape
    assert h > 0 and w > 0  # 크롭 결과가 있음


@pytest.mark.parametrize("pad_color", [(114, 114, 114), (0, 0, 0)])
def test_torch_crop_region_basic(pad_color):
    img = make_torch_image()
    poly = make_dummy_polygon()
    cropped = torch_crop_region(img, poly, pad_color=pad_color)
    assert isinstance(cropped, torch.Tensor)
    assert cropped.ndim == 3
    # pad_color가 적용된 픽셀 일부 확인
    mask = (cropped == torch.tensor(pad_color, dtype=cropped.dtype).view(3, 1, 1))
    assert mask.any()


def test_batch_crop_region_numpy():
    imgs = [make_numpy_image() for _ in range(2)]
    polys = [make_dummy_polygon() for _ in range(2)]
    outs = batch_crop_region(imgs, polys)
    assert isinstance(outs, list)
    assert isinstance(outs[0], np.ndarray)


def test_batch_crop_region_torch():
    imgs = [make_torch_image() for _ in range(2)]
    polys = [make_dummy_polygon() for _ in range(2)]
    outs = batch_crop_region(imgs, polys)
    assert isinstance(outs, list)
    assert isinstance(outs[0], torch.Tensor)


def test_batch_crop_region_invalid_input():
    with pytest.raises(TypeError):
        batch_crop_region(["not image"], [make_dummy_polygon()])
