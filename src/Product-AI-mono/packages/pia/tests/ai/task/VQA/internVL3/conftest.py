import gc
import os

import cv2
import numpy as np
import pytest
import torch
from pia.tests.test_config import (
    ASSETS_IMAGE_SAVE_DIR,
    ASSETS_MODEL_SAVE_DIR,
    ASSETS_VIDEO_SAVE_DIR,
)
from pia.utils.api.hugging_face import HFModelDownloader

HF_REPO_ID = "OpenGVLab/InternVL3-2B"
PEOPLE_IMAGE_FILE_NAME = os.path.join(ASSETS_IMAGE_SAVE_DIR, "people.jpeg")
PEOPLE_VIDEO_1_FILE_NAME = os.path.join(ASSETS_VIDEO_SAVE_DIR, "cat.mp4")
# PEOPLE_VIDEO_1_FILE_NAME = os.path.join(ASSETS_VIDEO_SAVE_DIR, "safety0001_clip.mp4")
PEOPLE_VIDEO_2_FILE_NAME = os.path.join(ASSETS_VIDEO_SAVE_DIR, "safety0001_clip.mp4")


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """각 테스트 후 GPU 메모리 정리"""
    yield

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"\n🧹 GPU memory cleaned. Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

@pytest.fixture
def model_path():
    return os.path.join(ASSETS_MODEL_SAVE_DIR, HF_REPO_ID)

@pytest.fixture
def img():
    img = cv2.imread(PEOPLE_IMAGE_FILE_NAME)
    if img is None:
        raise FileNotFoundError(f"Image not found at {PEOPLE_IMAGE_FILE_NAME}")
    return img

@pytest.fixture
def video():
    """단일 비디오를 (T, H, W, C) 형태의 numpy 배열로 로드"""
    cap = cv2.VideoCapture(PEOPLE_VIDEO_1_FILE_NAME)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found at {PEOPLE_VIDEO_1_FILE_NAME}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from {PEOPLE_VIDEO_1_FILE_NAME}")

    # (T, H, W, C) 형태로 반환
    video_array = np.array(frames)
    print(f"Loaded video shape: {video_array.shape}")
    return video_array

@pytest.fixture
def videos():
    """복수 비디오를 List[np.ndarray] 형태로 로드"""
    video_files = [PEOPLE_VIDEO_1_FILE_NAME, PEOPLE_VIDEO_2_FILE_NAME]
    all_videos = []

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found at {video_file}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames could be read from {video_file}")

        video_array = np.array(frames)
        print(f"Loaded video shape: {video_array.shape}")
        all_videos.append(video_array)

    return all_videos

@pytest.fixture(scope="module", autouse=True)
def download_models():
    downloader = HFModelDownloader()
    downloader.download(repo_id=HF_REPO_ID, save_dir=ASSETS_MODEL_SAVE_DIR, snapshot=True)
