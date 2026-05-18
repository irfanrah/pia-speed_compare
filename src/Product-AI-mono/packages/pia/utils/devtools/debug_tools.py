import functools
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2


def skip_in_debug_mode(func):
    """
    이 데코레이터는 debugpy 디버깅 모드에서 함수가 실행되지 않도록 합니다.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "PYTHONDEBUG" in os.environ:
            return None
        if "VSCODE_INSPECTOR_OPTIONS" in os.environ:
            return None
        if "PYDEVD_USE_FRAME_EVAL" in os.environ:
            return None
        return func(*args, **kwargs)

    return wrapper


def only_in_debug_mode(func):
    """
    이 데코레이터는 debugpy 디버깅 모드에서만 함수가 실행되도록 합니다.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "PYTHONDEBUG" in os.environ:
            return func(*args, **kwargs)
        if "VSCODE_INSPECTOR_OPTIONS" in os.environ:
            return func(*args, **kwargs)
        if "PYDEVD_USE_FRAME_EVAL" in os.environ:
            return func(*args, **kwargs)
        return None

    return wrapper


def try_except_only_in_prod_mode(func):
    """
    • 디버그 중이면: try/except 없이 실행 → 예외 그대로
    • 디버그 아니면: try/except 로 감싸 예외 무시 → None 반환
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        debug_active = (
            sys.gettrace() is not None  # 표준 디버거
            or "PYTHONDEBUG" in os.environ  # python -d
            or "VSCODE_INSPECTOR_OPTIONS" in os.environ  # VSCode debug
            or "PYDEVD_USE_FRAME_EVAL" in os.environ  # PyCharm/pydevd
        )
        if debug_active:  # ── 디버그: 예외 그대로
            return func(*args, **kwargs)

        try:  # ── 프로덕션: 예외 삼킴
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
    return wrapper


def check_debug_mode():
    if "PYTHONDEBUG" in os.environ:
        return True
    if "VSCODE_INSPECTOR_OPTIONS" in os.environ:
        return True
    if "PYDEVD_USE_FRAME_EVAL" in os.environ:
        return True
    return False


@only_in_debug_mode
def print_only_debug_mode(s):
    print(s)


def save_snapshot(image, save_dir="logs"):
    if image is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Asia/Seoul 시스템 시간 기준
        filename = os.path.join(save_dir, f"snapshot_{ts}.jpg")
        cv2.imwrite(filename, image)
