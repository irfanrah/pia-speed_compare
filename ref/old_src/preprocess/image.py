from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]


@torch.inference_mode()
def preprocess(
    img_path: str,
    batch: int,
    resize_size: Tuple[int, int] = (336, 336),
    device: str = "cuda",
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
) -> torch.Tensor:
    """Load a real image, normalize for PE-Core-L14-336, replicate to batch.

    Returns a CUDA tensor of shape (batch, 3, *resize_size), float32.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")

    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, copy=True)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = TF.resize(
        t,
        list(resize_size),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True,
    )
    t = TF.convert_image_dtype(t, torch.float32)
    t = TF.normalize(t, mean=mean, std=std)
    t = t.unsqueeze(0).repeat(batch, 1, 1, 1).contiguous()
    return t.to(device, non_blocking=True)
