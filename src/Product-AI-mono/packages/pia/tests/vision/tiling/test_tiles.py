import numpy as np
from pia.vision.tiling import Tile, tile_frame, tile_sequence, tile_videos

# @pytest.fixture
# def download_tile_samples():
#     gdrive_ids = {
#         "no_tile.png": "13a15TVHJUqmsfZuJFJebxGCtUWjhpftP",
#         "tile0.png": "1G43XMNFqAEU9nWRYokTzuOOsjDqmJDBW",
#         "tile1.png": "18o6WGaioHP1oFBzXoTB32od2swBoouhd",
#         "tile2.png": "1AurHUSoOZIDkWp-W-O9TFEzpN0ZoXcmY",
#         "tile3.png": "1_N2wrsvXtbINSkFppbtg53a_TjbZ5E-f",
#         "tile4.png": "1ZfxNnaoipdDHZlfwCU9de6yHqmNjvZPJ",
#         "tile5.png": "1MMPJ6K7OK_VOJk1V8z6-PyrhQ68tNLom",
#         "tile6.png": "1HAJvmjTqfael00aN8lKFAKflhCnLfLaZ",
#         "tile7.png": "1h8iuc6Hfpvo0Q-8bMHI_uYMviXJfMh6n",
#         "tile8.png": "15d6TzDJtFgtYmRD1tseKlkeda4yBofb7",
#     }

#     for filename, gdrive_id in gdrive_ids.items():
#         if os.path.exists(f"packages/pia/tests/sample_tiles/{filename}"):
#             continue

#         message = f"gdrive files download --destination packages/pia/tests/sample_tiles/ {gdrive_id}"
#         os.system(message)


# def test_tile_single_image(download_tile_samples):
#     image = cv2.imread("packages/pia/tests/sample_tiles/no_tile.png")
#     tiles = tile_frame(frame=image, tile_size=Tile("L"))

#     target_tile_files = natsorted(glob("packages/pia/tests/sample_tiles/tile*.png"))

#     target_tiles = []
#     for target_tile_file in target_tile_files:
#         if isinstance(target_tile_file, np.ndarray):
#             target_tile = target_tile_file
#         else:
#             target_tile = cv2.imread(target_tile_file)
#         target_tiles.append(target_tile)
#     target_tiles = np.asarray(target_tiles)

#     np_tiles = np.array(tiles)
#     assert (
#         np_tiles.shape == target_tiles.shape
#     ), "Shape mismatch: {tiles.shape} vs {target_tiles.shape}"
#     assert np.array_equal(np_tiles, target_tiles), "Tiles mismatch with target tiles"


def test_tile_frame():
    dummy_image = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    tiled_dummy_image = tile_frame(dummy_image, tile_size=Tile("L"), model_image_size=(448, 448))

    Tiles, Heights, Widths, Channels = np.array(tiled_dummy_image).shape

    assert Tiles == 9, "Tile count mismatch"  # tile_count is 19 if tile_size "S"
    assert Heights == 448, "Height mismatch"  # tile_size "S" H px is 448
    assert Widths == 448, "Width mismatch"  # tile_size "S" W px is 448
    assert Channels == 3, "Channel mismatch"


def test_tile_sequence():
    dummy_image = np.random.randint(
        0, 256, size=(3, 1080, 1920, 3), dtype=np.uint8
    )  # Batch, H, W, C
    tiled_dummy_image = tile_sequence(dummy_image, tile_size=Tile("S"), model_image_size=(448, 448))

    Batchs, Tiles, Heights, Widths, Channels = np.array(tiled_dummy_image).shape

    assert Batchs == 3, "Batch size mismatch"
    assert Tiles == 19, "Tile count mismatch"  # tile_count is 19 if tile_size "S"
    assert Heights == 448, "Height mismatch"  # tile_size "S" H px is 448
    assert Widths == 448, "Width mismatch"  # tile_size "S" W px is 448
    assert Channels == 3, "Channel mismatch"


def test_tile_video():
    dummy_image = np.random.randint(
        0, 256, size=(3, 12, 1080, 1920, 3), dtype=np.uint8
    )  # Batch, Sequence, H, W, C. It is a common shape for video input.
    tiled_dummy_image = tile_videos(dummy_image, tile_size="S", model_image_size=(448, 448))

    Batchs, Tiles, Sequences, Heights, Widths, Channels = np.array(tiled_dummy_image).shape

    assert Batchs == 3, "Batch size mismatch"
    assert Tiles == 19, "Tile count mismatch"  # tile_count is 19 if tile_size "S"
    assert Sequences == 12, "Sequence length mismatch"
    assert Heights == 448, "Height mismatch"  # tile_size "S" H px is 448
    assert Widths == 448, "Width mismatch"  # tile_size "S" W px is 448
    assert Channels == 3, "Channel mismatch"
