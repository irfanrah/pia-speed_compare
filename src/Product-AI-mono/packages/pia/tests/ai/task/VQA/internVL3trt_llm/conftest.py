"""Pytest fixtures for InternVL3 TRT-LLM VQA inference tests.

테스트용 이미지 로드와 프롬프트 생성 등 공통 준비 로직을 제공한다.
"""

import os

import cv2
import numpy as np
import pytest

from pia.tests.test_config import ASSETS_IMAGE_SAVE_DIR


IMAGE_300_PATH = os.path.join(ASSETS_IMAGE_SAVE_DIR, "300.png")
FALLDOWN_PROMPT = """
Analyze this image carefully. Determine if a person has fallen down.

Important classification rules:

- The "falldown" category applies to any person who is lying down, regardless of:
  - the surface (e.g., floor, mattress, bed)
  - the posture (natural or unnatural)
  - the cause (e.g., sleeping, collapsing, lying intentionally)
- This includes:
  - A person lying flat on the ground or other surfaces
  - A person collapsed or sprawled in any lying position
- The "normal" category applies only if the person is:
  - sitting
  - standing
  - kneeling
  - or otherwise upright (not lying down)

Answer in JSON format with BOTH of the following fields:
- "category": either "falldown" or "normal"
- "description": a brief reason why this classification was made
  (e.g., "person lying on a mattress", "person sitting on sofa")

Example:
{
  "category": "falldown",
  "description": "person lying on a mattress in natural posture"
}
"""


def _load_image(path: str) -> np.ndarray:
    """이미지를 로드하여 HWC numpy 배열로 반환한다."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def _make_red_image_like(img: np.ndarray) -> np.ndarray:
    """입력 이미지와 동일한 크기의 단색 빨간 이미지를 생성한다."""
    red = np.zeros_like(img)
    red[:, :, 2] = 255
    return red


@pytest.fixture(scope="session")
def image_300() -> np.ndarray:
    """테스트용 300.png 이미지를 로드한다(없으면 skip)."""
    if not os.path.isfile(IMAGE_300_PATH):
        pytest.skip("Test image not found")
    return _load_image(IMAGE_300_PATH)


@pytest.fixture(scope="session")
def red_image_300(image_300: np.ndarray) -> np.ndarray:
    """300.png와 같은 크기의 빨간색 이미지를 생성한다."""
    return _make_red_image_like(image_300)


@pytest.fixture(scope="session")
def falldown_prompt() -> str:
    """낙상 분류용 프롬프트를 제공한다."""
    return FALLDOWN_PROMPT
