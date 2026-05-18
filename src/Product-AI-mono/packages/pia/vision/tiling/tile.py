import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

tile_dict = {
    "no_tile": {"px": 10000, "ratio": 0.25},
    "L": {"px": 672, "ratio": 0.25},
    "M": {"px": 560, "ratio": 0.25},
    "S": {"px": 448, "ratio": 0.25},
}


@dataclass
class Tile:
    """
    Tile class representing a tile with a given size, pixel dimensions, and overlap ratio.
    The size must be one of "L", "M", or "S".

    Example:
        tile_s = Tile("S")
        print(tile_s)  # Output: Tile(size='S', px=448, overlap=0.25)
        print(tile_s.size)  # Output: 'S'
        print(tile_s.px)  # Output: 448
    """

    size: str
    px: int
    overlap: float

    def __init__(self, size: str):
        if not size:
            raise ValueError(
                f"Size cannot be None or empty. Must be one of {list(tile_dict.keys())}."
            )

        # If a Tile object is passed, extract its size
        if isinstance(size, Tile):
            size = size.size

        # Lookup in tile_dict and assign attributes
        if size in tile_dict:
            self.size = size
            self.px = tile_dict[size]["px"]
            self.overlap = tile_dict[size]["ratio"]
        else:
            raise ValueError(f"Invalid size '{size}'. Must be one of {list(tile_dict.keys())}.")

    def calc_tile_cnt(self, origin_size: int) -> int:
        """Calculate the number of tiles needed for a given dimension (height or width)."""
        min_margin = int(self.px * self.overlap)
        return math.ceil((origin_size - min_margin) / (self.px - min_margin))


def tile_videos(
    videos_sequenced: np.ndarray,
    tile_size: Literal["L", "M", "S"] = "S",
    model_image_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    if len(videos_sequenced.shape) != 5:
        raise ValueError(
            f"Video shape must be (Video batch size, Sequence length, Height, Width, Channel). Your current video shape is {videos_sequenced.shape}."
        )

    tile = Tile(tile_size)
    videos_tiled_sequenced = []
    for video in videos_sequenced:
        sequence_tiled = tile_sequence(
            sequence=video, tile_size=tile, model_image_size=model_image_size
        )
        tiles_sequenced = np.swapaxes(sequence_tiled, 0, 1)
        videos_tiled_sequenced.append(tiles_sequenced)

    return np.array(videos_tiled_sequenced)


def tile_sequence(
    sequence: np.ndarray,
    tile_size: Tile,
    model_image_size: tuple[int, int] = (224, 224),
) -> list[np.ndarray]:
    if len(sequence.shape) != 4:
        raise ValueError(
            f"Image shape must be (Image batch size, Height, Width, Channel). Your current image shape is {sequence.shape}."
        )

    sequence_tiled = []
    for frame in sequence:
        tiles = tile_frame(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            tile_size,
            model_image_size=model_image_size,
        )
        sequence_tiled.append(tiles)

    return sequence_tiled


def tile_frame(
    frame: np.ndarray, tile_size: Tile, model_image_size: tuple[int, int] = (224, 224)
) -> list[np.ndarray]:
    if len(frame.shape) != 3:
        raise ValueError(
            f"Image shape must be (Height, Width, Channel). Your current image shape is {frame.shape}."
        )

    img_h, img_w = frame.shape[:2]
    margin_size = int(tile_size.px * tile_size.overlap)
    step_size = tile_size.px - margin_size

    # Add original frame. It is resized to the tile size.
    tiled_images = [frame]

    # Check if tiling is necessary
    if step_size * 2 >= img_h and step_size * 2 >= img_w:
        return tiled_images

    # Calculate the number of tiles
    x_tile_cnt = tile_size.calc_tile_cnt(img_w)
    y_tile_cnt = tile_size.calc_tile_cnt(img_h)

    # Create tiles
    for y in range(y_tile_cnt):
        y_start = step_size * y if y != y_tile_cnt - 1 else img_h - tile_size.px
        y_end = min(img_h, y_start + tile_size.px)

        for x in range(x_tile_cnt):
            x_start = step_size * x if x != x_tile_cnt - 1 else img_w - tile_size.px
            x_end = min(img_w, x_start + tile_size.px)

            each_tile = frame[y_start:y_end, x_start:x_end]
            tiled_images.append(each_tile)

    resized_tiles = [
        cv2.resize(v, model_image_size, interpolation=cv2.INTER_LINEAR) for v in tiled_images
    ]

    return resized_tiles
