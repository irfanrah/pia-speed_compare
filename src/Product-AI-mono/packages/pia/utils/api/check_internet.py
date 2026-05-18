import functools
import socket

import pytest


def check_internet(host="8.8.8.8", port=53, timeout=3) -> bool:
    """인터넷 연결 여부 확인 (DNS 서버에 소켓 연결 시도)."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def require_internet(func):
    """인터넷 연결이 없으면 테스트를 skip 시키는 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not check_internet():
            pytest.skip("🌐 인터넷 연결이 없어 테스트를 건너뜁니다.")
        return func(*args, **kwargs)
    return wrapper
