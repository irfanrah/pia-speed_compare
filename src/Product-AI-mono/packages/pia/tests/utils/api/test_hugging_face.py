import functools
import os
import shutil
from pathlib import Path

import pytest
from pia.utils.api.check_internet import require_internet
from pia.utils.api.hugging_face import HFModelDownloader

MODEL_FILE = "dummy.pt"
REPO_ID = "dummy"
RAW_REPO_ID = str(Path(os.getenv("HF_NAMESPACE")) / Path(REPO_ID))

def require_token(func):
    """Hugging Face 토큰과 조직명이 없으면 테스트를 skip 시키는 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        token = os.getenv("HF_AUTH_TOKEN")
        org = os.getenv("HF_NAMESPACE")
        is_token = token is not None
        is_org = org is not None
        if not is_token or not is_org:
            message = f"token: {is_token}, org: {is_org}. 환경변수 HF_AUTH_TOKEN, HF_NAMESPACE 설정이 필요합니다."
            pytest.skip(f"Hugging Face 인증 정보가 없어 테스트를 건너뜁니다. {message}")
        return func(*args, **kwargs)
    return wrapper

def test_check_instance():
    downloader = HFModelDownloader()
    assert isinstance(downloader, HFModelDownloader), "Failed to create HFModelDownloader instance"


def test_downloader_extract_model_name_static_method(tmp_path: Path):
    downloader = HFModelDownloader()
    model_name = downloader.extract_model_name(path=MODEL_FILE)
    assert model_name == "dummy", f"Expected 'dummy', got '{model_name}'"

    model_name = downloader.extract_model_name(path= tmp_path / MODEL_FILE)
    assert model_name == "dummy", f"Expected 'dummy', got '{model_name}'"


@require_token
@require_internet
def test_download_method(tmp_path: Path):
    downloader = HFModelDownloader()
    downloader.download(
        repo_id=REPO_ID,
        save_dir=tmp_path,
        file_name=MODEL_FILE,
    )
    model_path = tmp_path / MODEL_FILE
    assert model_path.exists(), "Model file was not downloaded"
    assert model_path.stat().st_size > 0, "Downloaded file is empty"

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


@require_token
@require_internet
def test_download_file_method(tmp_path: Path):
    downloader = HFModelDownloader()
    downloader.download_file(
        repo_id=RAW_REPO_ID,
        file_name=MODEL_FILE,
        save_dir=tmp_path,
    )
    model_path = tmp_path / MODEL_FILE
    assert model_path.exists(), "Model file was not downloaded"
    assert model_path.stat().st_size > 0, "Downloaded file is empty"

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


@require_token
@require_internet
def test_download_snapshot_method(tmp_path: Path):
    downloader = HFModelDownloader()
    downloader.download_snapshot(
        repo_id=RAW_REPO_ID,
        save_dir=tmp_path,
    )
    model_path = tmp_path / Path(RAW_REPO_ID) / MODEL_FILE
    assert model_path.exists(), "Model file was not downloaded"
    assert model_path.stat().st_size > 0, "Downloaded file is empty"

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()
