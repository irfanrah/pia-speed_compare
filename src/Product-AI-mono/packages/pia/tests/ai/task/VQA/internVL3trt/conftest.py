import os
import cv2
import numpy as np
import pytest
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.VQA.base import VQAConfig
from pia.tests.test_config import (
    ASSETS_IMAGE_SAVE_DIR,
    ASSETS_VIDEO_SAVE_DIR,
)

PEOPLE_IMAGE_FILE_NAME = os.path.join(ASSETS_IMAGE_SAVE_DIR, "289.jpg")
IMAGE_300_FILE_NAME = os.path.join(ASSETS_IMAGE_SAVE_DIR, "300.png")
PEOPLE_VIDEO_1_FILE_NAME = os.path.join(ASSETS_VIDEO_SAVE_DIR, "cat.mp4")
PEOPLE_VIDEO_2_FILE_NAME = os.path.join(ASSETS_VIDEO_SAVE_DIR, "elevator_violence_054.mp4")

@pytest.fixture(scope="session")
def blue_image():
    """파란색으로 채워진 가상 이미지를 BGR 형식으로 생성합니다."""
    # BGR 형식에서 파란색은 (255, 0, 0) 입니다. (Blue, Green, Red)
    img = np.full((224, 224, 3), (255, 0, 0), dtype=np.uint8)
    return img


@pytest.fixture(scope="session")
def big_blue_image():
    """파란색으로 채워진 큰 가상 이미지를 BGR 형식으로 생성합니다."""
    # BGR 형식에서 파란색은 (255, 0, 0) 입니다.
    img = np.full((1080, 1920, 3), (255, 0, 0), dtype=np.uint8)
    return img


@pytest.fixture(scope="session")
def red_image():
    """붉은색으로 채워진 가상 이미지를 BGR 형식으로 생성합니다."""
    # BGR 형식에서 붉은색은 (0, 0, 255) 입니다. (Blue, Green, Red)
    img = np.full((224, 224, 3), (0, 0, 255), dtype=np.uint8)
    return img


@pytest.fixture
def img():
    img = cv2.imread(PEOPLE_IMAGE_FILE_NAME)
    if img is None:
        raise FileNotFoundError(f"Image not found at {PEOPLE_IMAGE_FILE_NAME}")
    return img


@pytest.fixture(scope="session")
def image_300():
    img = cv2.imread(IMAGE_300_FILE_NAME)
    if img is None:
        raise FileNotFoundError(f"Image not found at {IMAGE_300_FILE_NAME}")
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





@pytest.fixture(scope="session")
def rainbow_video():
    """12프레임의 무지개색 비디오를 생성합니다. (T,H,W,C) 형태"""
    # 무지개색 정의 (BGR 형식)
    rainbow_colors = [
        (0, 0, 255),      # 빨강
        (0, 127, 255),    # 주황
        (0, 255, 255),    # 노랑
        (0, 255, 0),      # 초록
        (255, 255, 0),    # 청록
        (255, 127, 0),    # 하늘색
        (255, 0, 0),      # 파랑
        (255, 0, 127),    # 남색
        (255, 0, 255),    # 보라
        (127, 0, 255),    # 자홍
        (0, 0, 255),      # 다시 빨강으로
        (0, 255, 255),    # 노랑으로 마무리
    ]

    # (T,H,W,C) 형태의 비디오 생성
    video = np.zeros((12, 448, 448, 3), dtype=np.uint8)

    for frame_idx, color in enumerate(rainbow_colors):
        video[frame_idx] = np.full((448, 448, 3), color, dtype=np.uint8)

    return video

@pytest.fixture(scope="session")
def gradient_video():
    """그라데이션 효과의 12프레임 비디오를 생성합니다. (T,H,W,C) 형태"""
    video = np.zeros((12, 448, 448, 3), dtype=np.uint8)

    for frame_idx in range(12):
        for y in range(224):
            for x in range(224):
                r = int((frame_idx / 11) * 255)
                g = int((x / 223) * 255)
                b = int((y / 223) * 255)
                video[frame_idx, y, x] = [b, g, r]

    return video

@pytest.fixture(scope="session")
def text_overlay_video():
    """프레임 번호가 표시된 12프레임 비디오를 생성합니다."""
    video = np.zeros((12, 448, 448, 3), dtype=np.uint8)

    for frame_idx in range(12):
        bg_color = (
            int(255 * (frame_idx / 11)),  # B
            int(255 * (1 - frame_idx / 11)),  # G
            128  # R (고정)
        )
        video[frame_idx] = np.full((448, 448, 3), bg_color, dtype=np.uint8)

        frame_with_text = video[frame_idx].copy()
        text = f"Frame {frame_idx + 1}"
        cv2.putText(frame_with_text, text, (50, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video[frame_idx] = frame_with_text

    return video

@pytest.fixture(scope="session")
def vqa_config():
    """테스트용 VQAConfig 객체를 생성합니다."""
    model_path = None
    return VQAConfig(
        model_path=model_path,
        device="cuda",
        frame_per_tile_max_num=16,
        data_type="bfloat16",
        n_gpus=1,
        save_cache_dir=model_path,
        api_host="127.0.0.1",
        api_port=9997,
        max_new_tokens= 14
    )

@pytest.fixture(scope="session")
def vqa_model(vqa_config):
    """테스트 세션 동안 한 번만 모델을 로드합니다."""
    print("\n--- Loading VQA Model for session ---")
    model = PiaTorchModel(target_task="VQA", target_model="internvl3trt", config=vqa_config)
    return model
