import logging
import sys

import pytest

# 실제 경로로 수정
from pia.utils.logging.logger import CallLog


def test_call_level_mapping_str_and_int():
    assert CallLog._call_level("debug") == logging.DEBUG
    assert CallLog._call_level("Degub") == logging.DEBUG
    assert CallLog._call_level("INFO") == logging.INFO
    assert CallLog._call_level(CallLog.WARNING) == logging.WARNING


def test_add_file_handler_creates_file_and_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    logdir = tmp_path / "logs"
    logdir.mkdir()
    c = CallLog(log_name="test.add_file", log_level="debug")

    # 파일 핸들러 추가 (TimedRotating)
    base = logdir / "testlog"
    c.add_file_handler(path=str(base), H_type="R", log_level="debug")

    # 핸들러 타입/이름 확인
    fh = c._file_handler
    from logging.handlers import TimedRotatingFileHandler
    assert isinstance(fh, TimedRotatingFileHandler)
    assert fh.name == "file_handler"

    # 파일이 실제로 생성되었는지 확인
    files = list(logdir.glob("testlog*" + c._log_extension))
    assert len(files) == 1
    logfile = files[0]
    assert logfile.exists()

    # 로그 한 줄 쓰고 포맷 확인
    c.info("hello file")
    fh.flush()
    text = logfile.read_text(encoding="utf-8")
    # 파일 포맷: "%(asctime)s/%(levelname)s\n%(message)s"
    assert "/INFO\n" in text
    assert "hello file" in text

    # 정리
    c.remove_file_handler()
    assert fh not in c._logger.handlers
    assert fh.stream is None  # 닫혔는지 확인


def test_add_stream_handler_stream_and_levels(capsys):
    c = CallLog(log_name="test.stream", log_level="info")
    c.add_stream_handler(H_type="stream", log_level="warning")

    # info는 출력되지 않고 warning만 출력되어야 함
    c.info("info-msg")
    c.warning("warn-msg")

    out = capsys.readouterr()
    stream = out.err + out.out  # 환경에 따라 err/out 어느 쪽으로 갈 수 있음
    assert "info-msg" not in stream
    assert "warn-msg" in stream


def test_add_stream_handler_rich_type():
    c = CallLog(log_name="test.rich", log_level="debug")
    c.add_stream_handler(H_type="rich", log_level="debug")
    # RichHandler 타입 확인 (import 경로에 따라 비교)
    from rich.logging import RichHandler
    assert isinstance(c._stream_handler, RichHandler)
    assert c._stream_handler.name == "stream_handler"


def test_handler_level_assertion():
    # 루트 WARNING(30)에서 핸들러를 DEBUG(10)로 낮추면 assert 터져야 함
    c = CallLog(log_name="test.level.assert", log_level="warning")
    with pytest.raises(AssertionError):
        c.add_stream_handler(H_type="stream", log_level="debug")


def test_handle_exception_three_tuple(caplog):
    c = CallLog(log_name="test.exc", log_level="debug", propagate=True)
    c.add_stream_handler(H_type="stream", log_level="debug")

    with caplog.at_level(logging.ERROR, logger=c._logger.name):
        try:
            raise ValueError("boom!")
        except Exception:
            et, ev, tb = sys.exc_info()
            c.handle_exception(et, ev, tb)

    assert any("Unexpected exception" in r.message for r in caplog.records)
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_handle_exception_sys_excepthook_like(caplog):
    c = CallLog(log_name="test.exc2", log_level="debug", propagate=True)
    c.add_stream_handler(H_type="stream", log_level="debug")

    class EHArgs:
        def __init__(self, exc_type, exc_value, exc_tb):
            self.exc_type = exc_type
            self.exc_value = exc_value
            self.exc_traceback = exc_tb

    with caplog.at_level(logging.ERROR, logger=c._logger.name):
        try:
            {}["x"]  # KeyError 발생
        except Exception:
            et, ev, tb = sys.exc_info()
            c.handle_exception(EHArgs(et, ev, tb))

    assert any("Unexpected exception" in r.message for r in caplog.records)
