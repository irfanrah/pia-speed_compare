import numpy as np
import torch


def raise_exception_decorator(exception):
    def decorator(function):
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except exception:
                if exception == FileNotFoundError:
                    raise exception(f"Check model File Path / ERROR : {exception.__name__}")

        return wrapper

    return decorator


def validate_tile_config(tile_config: dict):
    required_keys = {"size", "overlap_ratio"}
    if tile_config.keys() != required_keys:
        raise ValueError(
            f"`tile_config` must have `size` and `overlap_ratio` as its keys. Your current `tile_config`s keys are {tile_config.keys()}."
        )

    if not isinstance(tile_config["size"], int):
        raise ValueError(
            f"`tile_config['size']` must be int type. Your current `tile_config['size'] is {tile_config['size']}."
        )

    if not isinstance(tile_config["overlap_ratio"], float):
        raise ValueError(
            f"`tile_config['overlap_ratio']` must be float type. Your current `tile_config['overlap_ratio'] is {tile_config['overlap_ratio']}."
        )
    if tile_config["overlap_ratio"] < 0.0 or tile_config["overlap_ratio"] > 1.0:
        raise ValueError(
            f"`tile_config['overlap_ratio]` must be from 0.0 to 1.0. Your current `tile_config['overlap_ratio']` is {tile_config['overlap_ratio']}."
        )


def validate_video_shape(video: np.ndarray):
    if len(video.shape) not in (4, 5):
        raise ValueError(
            f"Video needs 4 or 5 dimension, [frame, height, width, channel] or [batch, frame, height, width, channel]. Your current video shape is {video.shape}"
        )


def validate_text_embedding_vector(text_embedding_vector: torch.Tensor):
    if len(text_embedding_vector.shape) != 2:
        raise ValueError(
            f"The shape of text embedding vector must be (Number of captions, Embedding dimension). The shape of your text embedding vector is {text_embedding_vector.shape}"
        )


def validate_video_embedding_vector(video_embedding_vector: torch.Tensor):
    if len(video_embedding_vector.shape) != 3:
        raise ValueError(
            f"The shape of video embedding vector must be (Batch size, Number of tiles, Embedding dimension). The shape of your video embedding vector is {video_embedding_vector.shape}"
        )
