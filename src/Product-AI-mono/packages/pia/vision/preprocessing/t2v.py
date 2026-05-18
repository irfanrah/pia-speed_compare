from typing import Literal, Tuple

import cv2
import numpy as np
import torch
from pia.utils.exception.model_handler import validate_video_shape
from pia.vision.tiling import tile_videos
from PIL import Image
from torchvision.transforms import Compose, transforms


def video_preprocess(
    video: np.ndarray,
    device: Literal["cuda", "cpu", "mps"] = None,
    tile_size: Literal[None, "L", "M", "S"] = None,
    img_size: Tuple[int, int] = (224, 224),
    transform: transforms.Compose = Compose([]),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses the video by tiling and resizing it.

    Args:
        video (np.ndarray): Video array. Shape: (Batch size, Sequence length, Height, Width, Channel).
        device (str, optional): Device. "cuda" or "cpu" or "mps". Defaults to None.
        tile_size (str, optional): Tile size. "L" or "M" or "S". Defaults to None.
        img_size (Tuple[int, int], optional): Image size after resizing. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transformations to apply to each frame. Defaults to Compose().

    Returns:
        video_tensor (torch.Tensor): Preprocessed video tensor. Shape: (Batch size, Number of tiles, Sequence length, Channel, Height, Width).
        video_mask_tensor (torch.Tensor): Video mask tensor. Shape: (Batch size, Number of tiles, Sequence length).
    """

    # TODO : numpy, tensor 둘다 가능한지 확인 필요
    validate_video_shape(video)
    device = device if device else "cpu"

    has_batch_dim = True if len(video.shape) == 5 else False
    has_tile_config = True if tile_size else False

    if not has_batch_dim:
        video = video[None, ::]
        has_batch_dim = True

    if has_tile_config:
        tiled_video = tile_videos(videos_sequenced=video, tile_size=tile_size)
    else:
        tiled_video = np.expand_dims(video, axis=1)

    batch_size, num_tiles, sequence_length, height, width, channel = tiled_video.shape
    tiled_and_reshaped_video = tiled_video.reshape(
        batch_size * num_tiles * sequence_length, height, width, channel
    )

    if num_tiles == 1:
        resized_video = [
            cv2.resize(v, img_size, interpolation=cv2.INTER_LINEAR) for v in tiled_and_reshaped_video
        ]

    else:
        resized_video = tiled_and_reshaped_video

    preprocessed_video_tensor = torch.stack(
        [transform(v) for v in resized_video],
        dim=0,
    )

    resize_target_height = img_size[1]
    resize_target_width = img_size[0]

    video_tensor = preprocessed_video_tensor.reshape(
        (
            batch_size,
            num_tiles,
            sequence_length,
            channel,
            resize_target_height,
            resize_target_width,
        )
    ).to(device)

    video_mask_tensor = torch.ones(
        batch_size, num_tiles, sequence_length, dtype=torch.float32, device=device
    )

    return video_tensor, video_mask_tensor


def PE_video_preprocess(
    video: np.ndarray,
    device: Literal["cuda", "cpu", "mps"] = None,
    tile_size: Literal[None, "L", "M", "S"] = None,
    img_size: Tuple[int, int] = (None, None),
    transform: transforms.Compose = Compose([]),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses the video by tiling and resizing it.

    Args:
        video (np.ndarray): Video array. Shape: (Batch size, Sequence length, Height, Width, Channel).
        device (str, optional): Device. "cuda" or "cpu" or "mps". Defaults to None.
        tile_size (str, optional): Tile size. "L" or "M" or "S". Defaults to None.
        img_size (Tuple[int, int], optional): Image size after resizing. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transformations to apply to each frame. Defaults to Compose().

    Returns:
        video_tensor (torch.Tensor): Preprocessed video tensor. Shape: (Batch size, Number of tiles, Sequence length, Channel, Height, Width).
        video_mask_tensor (torch.Tensor): Video mask tensor. Shape: (Batch size, Number of tiles, Sequence length).
    """

    # TODO : Tile version is not implemented
    validate_video_shape(video)
    device = device if device else "cpu"

    has_batch_dim = True if len(video.shape) == 5 else False
    has_tile_config = True if tile_size else False

    if not has_batch_dim:
        video = video[None, ::]
        has_batch_dim = True

    if has_tile_config:
        tiled_video = tile_videos(videos_sequenced=video, tile_size=tile_size)
    else:
        tiled_video = np.expand_dims(video, axis=1)

    batch_size, num_tiles, sequence_length, height, width, channel = tiled_video.shape
    tiled_and_reshaped_video = tiled_video.reshape(
        batch_size * num_tiles * sequence_length, height, width, channel
    )
    # Assert image size shoule be None or (depends on model)
    # if num_tiles == 1:
    #     resized_video = [
    #         cv2.resize(v, img_size, interpolation=cv2.INTER_LINEAR)
    #         for v in tiled_and_reshaped_video
    #     ]

    # else:
    #     resized_video = tiled_and_reshaped_video

    transformed = torch.stack(
        [transform(Image.fromarray(frame)) for frame in tiled_and_reshaped_video]
    )  # -> (B*T*S, C, H_new, W_new)

    # print(f"Transformed shape: {transformed.shape}")
    C_t, H_t, W_t = transformed.shape[1:]

    preprocessed_video_tensor = transformed.reshape(
        batch_size * num_tiles, sequence_length, C_t, H_t, W_t
    )

    # resize_target_height = img_size[1]
    # resize_target_width = img_size[0]

    # video_tensor = preprocessed_video_tensor.reshape(
    #     (
    #         batch_size,
    #         num_tiles,
    #         sequence_length,
    #         channel,
    #         resize_target_height,
    #         resize_target_width,
    #     )
    # ).to(device)
    video_tensor = preprocessed_video_tensor.to(device)

    video_mask_tensor = torch.ones(
        batch_size, num_tiles, sequence_length, dtype=torch.float32, device=device
    )

    return video_tensor, video_mask_tensor
