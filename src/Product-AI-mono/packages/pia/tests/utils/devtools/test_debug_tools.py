import sys

import pytest

# 실제 경로로 수정하세요
from pia.utils.devtools.debug_tools import (
    check_debug_mode,
    only_in_debug_mode,
    print_only_debug_mode,
    skip_in_debug_mode,
    try_except_only_in_prod_mode,
)


def _clear_debug_env(monkeypatch):
    monkeypatch.delenv("PYTHONDEBUG", raising=False)
    monkeypatch.delenv("VSCODE_INSPECTOR_OPTIONS", raising=False)
    monkeypatch.delenv("PYDEVD_USE_FRAME_EVAL", raising=False)


def test_skip_in_debug_mode(monkeypatch):
    # 디버그 모드 OFF
    _clear_debug_env(monkeypatch)

    called = {"n": 0}

    @skip_in_debug_mode
    def f(x):
        called["n"] += 1
        return x * 2

    assert f(3) == 6
    assert called["n"] == 1

    # 디버그 모드 ON
    monkeypatch.setenv("PYTHONDEBUG", "1")
    assert f(3) is None
    assert called["n"] == 1  # 호출 횟수 증가 없음


def test_only_in_debug_mode(monkeypatch, capsys):
    # 디버그 모드 OFF
    _clear_debug_env(monkeypatch)

    called = {"n": 0}

    @only_in_debug_mode
    def g(msg):
        called["n"] += 1
        print(msg)

    assert g("hello") is None
    assert called["n"] == 0
    out = capsys.readouterr().out
    assert "hello" not in out

    # 디버그 모드 ON
    monkeypatch.setenv("VSCODE_INSPECTOR_OPTIONS", "1")
    assert g("world") is None  # print 함수 자체는 None 반환
    assert called["n"] == 1
    out = capsys.readouterr().out
    assert "world" in out


def test_try_except_only_in_prod_mode_raises_in_debug(monkeypatch):
    # PROD 모드 OFF
    _clear_debug_env(monkeypatch)
    monkeypatch.setattr(sys, "gettrace", lambda: object())  # 디버거 활성화 시뮬레이션

    # PROD 모드 ON
    @try_except_only_in_prod_mode
    def boom():
        raise ValueError("boom!")

    with pytest.raises(ValueError, match="boom!"):
        boom()


def test_try_except_only_in_prod_mode_swallows_in_prod(monkeypatch, capsys):
    # PROD 모드 OFF
    _clear_debug_env(monkeypatch)
    monkeypatch.setattr(sys, "gettrace", lambda: None)

    # PROD 모드 ON
    @try_except_only_in_prod_mode
    def boom():
        raise RuntimeError("prod error")

    ret = boom()
    assert ret is None
    out = capsys.readouterr().out
    assert "prod error" in out


def test_try_except_only_in_prod_mode_ok_return(monkeypatch):
    # PROD 모드 OFF
    _clear_debug_env(monkeypatch)
    monkeypatch.setattr(sys, "gettrace", lambda: None)

    # PROD 모드 ON
    @try_except_only_in_prod_mode
    def ok(x):
        return x + 1

    assert ok(41) == 42


def test_check_debug_mode_env(monkeypatch):
    # check_debug_mode는 env만 보고 판단(현재 구현 기준)
    _clear_debug_env(monkeypatch)
    assert check_debug_mode() is False

    monkeypatch.setenv("PYTHONDEBUG", "1")
    assert check_debug_mode() is True

    _clear_debug_env(monkeypatch)
    monkeypatch.setenv("PYDEVD_USE_FRAME_EVAL", "1")
    assert check_debug_mode() is True


def test_print_only_debug_mode(monkeypatch, capsys):
    # 디버그 모드 OFF
    _clear_debug_env(monkeypatch)
    print_only_debug_mode("not shown")
    out = capsys.readouterr().out
    assert "not shown" not in out

    # 디버그 모드 ON
    monkeypatch.setenv("PYTHONDEBUG", "1")
    print_only_debug_mode("shown")
    out = capsys.readouterr().out
    assert "shown" in out
