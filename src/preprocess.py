"""Image preprocessing for PE-Core-L14-336.

Stage in the upstream pipeline: ``model_preprocess`` (BGR→RGB, resize, scale
to [0,1], normalize with mean/std=0.5). Mirrors
``pia_prod.AI.modules.perception_encoder.trt_utils.preprocess_image`` but is
standalone (no pia_prod dependency).
"""

from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

PE_MEAN = [0.5, 0.5, 0.5]
PE_STD = [0.5, 0.5, 0.5]


def load_image(img_path: str) -> np.ndarray:
    """Load an image from disk as a uint8 RGB ndarray (HWC)."""
    return np.array(Image.open(img_path).convert("RGB"), copy=True)


@torch.inference_mode()
def preprocess(
    images: Union[np.ndarray, List[np.ndarray]],
    resize_size: Tuple[int, int] = (336, 336),
    device: str = "cuda",
    mean=PE_MEAN,
    std=PE_STD,
) -> torch.Tensor:
    """Convert HWC uint8 RGB image(s) to a normalized BCHW float32 CUDA tensor.

    Accepts a single HWC ndarray or a list of HWC ndarrays. Returns shape
    ``(B, 3, *resize_size)``.
    """
    if isinstance(images, np.ndarray):
        images = [images]
    if len(images) == 0:
        raise ValueError("images must contain at least one frame")

    tensors = []
    for img in images:
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        t = TF.resize(
            t,
            list(resize_size),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensors.append(t)
    x = torch.stack(tensors, dim=0)
    x = TF.convert_image_dtype(x, torch.float32)
    x = TF.normalize(x, mean=mean, std=std)
    return x.to(device, non_blocking=True).contiguous()
