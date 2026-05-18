import numpy as np
import pytest
import torch
from pia.vision.preprocessing import LetterBox, LetterBoxTorch, resize_batches


def make_numpy_images(batch_size=2, h=240, w=320):
    """더미 np.ndarray 이미지 배치 생성"""
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(batch_size)]


def make_torch_images(batch_size=2, h=240, w=320, device="cpu"):
    """더미 torch.Tensor 이미지 배치 생성 (C,H,W)"""
    return [torch.randint(0, 256, (3, h, w), dtype=torch.uint8, device=device) for _ in range(batch_size)]


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_letterboxtorch_on_torch_images(device):
    imgs = make_torch_images(batch_size=2, h=240, w=320, device=device)
    lb = LetterBoxTorch(max_batch=2, target_size=(640, 640), device=device, dtype=torch.uint8)

    out = lb(imgs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 640, 640)
    # 패딩 영역이 pad_color로 채워졌는지 확인
    pad_val = lb.pad_color[0]
    assert (out[:, :, 0, 0] == pad_val).all()


def test_letterbox_on_numpy_images():
    imgs = make_numpy_images(batch_size=2, h=240, w=320)
    lb = LetterBox(new_shape=(640, 640))

    results = [lb(image=img) for img in imgs]
    assert isinstance(results, list)
    assert results[0].shape[:2] == (640, 640)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_resize_batches_with_torch_list(dtype):
    imgs = make_torch_images(batch_size=2, h=240, w=320)
    imgs = [im.to(dtype=dtype) for im in imgs]

    lb = LetterBoxTorch(max_batch=2, target_size=(640, 640), device="cpu", dtype=torch.uint8)
    processed, resized = resize_batches(imgs, device="cpu", letterbox_instance=lb)

    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (2, 3, 640, 640)
    assert isinstance(resized, torch.Tensor)


def test_resize_batches_with_numpy_list():
    imgs = make_numpy_images(batch_size=2, h=240, w=320)
    lb = LetterBox(new_shape=(640, 640))

    processed, resized = resize_batches(imgs, device="cpu", letterbox_instance=lb, shape=(640, 640))
    assert isinstance(processed, torch.Tensor)
    assert isinstance(resized, np.ndarray)
    assert processed.shape[1:] == (3, 640, 640)
    assert resized[0].shape[:2] == (640, 640)
