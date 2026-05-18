import os

ASSETS_SAVE_DIR = "assets"
ASSETS_MODEL_SAVE_DIR = os.path.join(ASSETS_SAVE_DIR, "models")
ASSETS_VIDEO_SAVE_DIR = os.path.join(ASSETS_SAVE_DIR, "videos")
ASSETS_IMAGE_SAVE_DIR = os.path.join(ASSETS_SAVE_DIR, "images")
ASSETS_ANNOTATION_SAVE_DIR = os.path.join(ASSETS_SAVE_DIR, "annotations")
TEST_NAME_SPACE = "PIA-SPACE-LAB"

TEST_VIDEOS = [
    "safety0001.mp4",
    "safety0002.mp4",
    "safety0003.mp4",
    "safety0001_clip.mp4",
    "safety0002_clip.mp4",
    "safety0003_clip.mp4",
    "cat.mp4",
    "elevator_violence_054.mp4"
]
TEST_IMAGES = [
    "people.jpeg",
    "290.png",
    "289.jpg",
    "300.png",
]
