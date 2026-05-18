"""
HD(1280x720) 멀티채널 GPU 한계 테스트

목적: CCTV 다채널 동시 처리 시 GPU 메모리/지연시간 한계 파악
실행: pytest -s packages/pia/tests/ai/task/CP/models/clip_ebc/test_limit_channel_hd.py
"""

import pytest

from pia.tests.ai.task.CP.models.clip_ebc._channel_limit_helper import (
    create_model_fixture,
    run_channel_limit_test,
)

HD_H, HD_W = 720, 1280


@pytest.fixture(scope="module")
def model():
    return create_model_fixture(HD_H, HD_W)


def test_hd_max_channel_limit(model):
    """
    [TC] HD 멀티채널 GPU 한계 탐색
    - 채널 수를 1 -> 2 -> 4 -> ... -> 1024 로 증가
    - 각 채널 수에서: 지연시간, 총 타일 수, GPU/CPU 메모리 측정
    - OOM 또는 예외 발생 시 중단하고 최대 처리 가능 채널 수 보고
    """
    run_channel_limit_test(model, HD_H, HD_W, f"HD({HD_W}x{HD_H})")
