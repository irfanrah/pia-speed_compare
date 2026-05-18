from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
import torch
from pia.ai.device import load_model_backend
from torchvision.transforms.functional import resize as tf_resize


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(
        self,
        new_shape=(640, 640),
        padding_color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.padding_color = padding_color

    def __call__(self, image=None, labels=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (
                labels["ratio_pad"],
                (dw, dh),
            )  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class LetterBoxTorch:
    def __init__(
        self,
        max_batch: int,
        target_size: tuple[int, int],
        pad_color: tuple[int, int, int] = (114, 114, 114),
        device: str = "cuda",
        dtype: torch.dtype = torch.uint8,
    ):
        """
        max_batch: 초기 최대 배치 크기 (필요 시 자동 확장)
        target_size: (target_h, target_w)
        pad_color: (R,G,B)
        """
        self.th, self.tw = target_size
        self.device = device
        self.dtype = dtype
        self.pad_color = pad_color

        self.pad_color_tensor = (
            torch.tensor(pad_color, dtype=dtype, device=device).view(1, 3, 1, 1)
        )

        # 초기 out 버퍼
        self._alloc_out(max_batch)

    def _alloc_out(self, batch_size: int):
        """버퍼 새로 할당"""
        self.max_batch = batch_size
        self.out = torch.empty(
            (batch_size, 3, self.th, self.tw),
            dtype=self.dtype,
            device=self.device,
        )

    def __call__(self, imgs: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        imgs: (B, C, H, W)
        return: (B, C, th, tw)
        """
        b = len(imgs)

        # 필요 시 out 버퍼 확장
        if b > self.max_batch:
            self._alloc_out(b)

        self.out[:b] = self.pad_color_tensor.expand(b, -1, self.th, self.tw)
        for i, im in enumerate(imgs):
            _, h, w = im.shape
            # aspect 유지한 resize 크기
            scale = min(self.th / h, self.tw / w)
            nh, nw = int(h * scale), int(w * scale)

            # 이미지 한장씩 resize
            resized = tf_resize(im, (nh, nw))

            # 중앙 정렬 위치
            pad_top = (self.th - nh) // 2
            pad_left = (self.tw - nw) // 2

            # 붙여넣기
            self.out[i, :, pad_top:pad_top + nh, pad_left:pad_left + nw] = torch.flip(resized, dims=[0])

        return self.out[:b]


def letterboxing_frame(im, letterbox_instance):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
    Returns:
        (list): A list of transformed images.
    """
    return [letterbox_instance(x) for x in im]


def preprocess_images(ims, device, half=False, letterbox_instance=None, shape=None):
    # TODO : im -> ims로 변경 필요 단수 아니고 복수로 입력되고 있음
    """
    Preprocesses an image for inference.

    Args:
        im (torch.Tensor or numpy.ndarray):
            The input image.
        device (torch.device):
            The device to use for computation.
        half (bool, optional):
            Whether to convert the image to half-precision floating point. Defaults to False.
        shape (list, optional):
            The desired shape of the image. Defaults to [640, 640].
    Returns:
        torch.Tensor: The preprocessed image.
    """
    not_tensor = not isinstance(ims, torch.Tensor)
    if not_tensor:
        if letterbox_instance is None and shape is not None:
            letter_im = np.stack([cv2.resize(img, shape) for img in ims])
        else:
            letter_im = np.stack(letterboxing_frame(ims, letterbox_instance=letterbox_instance))
        ims = letter_im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        ims = np.ascontiguousarray(ims)  # contiguous
        ims = torch.from_numpy(ims)

    ims = ims.to(device)
    ims = ims.half() if half else ims.float()  # uint8 to fp16/32
    if not_tensor:
        ims /= 255  # 0 - 255 to 0.0 - 1.0
    return ims, letter_im


def gpu_letterbox(batches: Union[List[torch.Tensor], torch.Tensor], letterbox_instance: LetterBoxTorch) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = batches[0].dtype  # 입력 이미지의 dtype과 동일하게 진행 float16 or float32
    resized_frames = letterbox_instance(imgs=batches)  # torch.uint8
    processed_frames = resized_frames.to(dtype) / 255  # uint8, float32 dtype -> float32, float16 -> float16
    return processed_frames, resized_frames


def cpu_letterbox(batches, device="cpu", letterbox_instance: LetterBox = None, shape=None) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    processed_frames, resized_frames = preprocess_images(
        batches,
        device=load_model_backend(device),
        letterbox_instance=letterbox_instance,
        shape=shape
    )
    return processed_frames, resized_frames


def resize_batches(batches, device="cpu", letterbox_instance: Union[LetterBox, LetterBoxTorch] = None, shape=None) :
    if (isinstance(batches, list) and all(isinstance(x, torch.Tensor) for x in batches)) or \
            (isinstance(batches, torch.Tensor)):
        processed_frames, resized_frames = gpu_letterbox(
            batches,
            letterbox_instance=letterbox_instance)
    elif (isinstance(batches, list) and all(isinstance(x, np.ndarray) for x in batches)) or \
            (isinstance(batches, np.ndarray)):
        processed_frames, resized_frames = cpu_letterbox(
            batches=batches,
            device=device,
            letterbox_instance=letterbox_instance,
            shape=shape
        )
    else :
        raise ValueError(f"Wrong batch type : {type(batches)}")
    return processed_frames, resized_frames
