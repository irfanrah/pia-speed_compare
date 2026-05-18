# tests/test_datalake_manager.py
import io

import boto3
import numpy as np
import pytest

# 실제 모듈 경로로 바꿔주세요
# 예: from pia_prod.AI.utils.data_lake_manager import DataLakeManager
from pia.utils.api.datalake import DataLakeManager
from PIL import Image


@pytest.fixture
def fake_boto3(monkeypatch):
    """boto3.session.Session을 가짜로 바꿔 네트워크 호출 없이 테스트."""

    class FakeClient:
        def __init__(self):
            self.calls = []

        def put_object(self, **kwargs):
            self.calls.append(kwargs)

    class FakeSession:
        def __init__(self, *args, **kwargs):
            self._client = FakeClient()

        def client(self, *args, **kwargs):
            return self._client

    # boto3.session.Session 교체
    monkeypatch.setattr(boto3.session, "Session", FakeSession)
    return FakeSession


def _last_put_object_kwargs(manager):
    """FakeClient의 마지막 put_object 인자를 가져옴."""
    fake_client = manager.client
    assert fake_client.calls, "put_object가 호출되지 않았습니다."
    return fake_client.calls[-1]


def test_upload_file(fake_boto3):
    # Arrange
    mgr = DataLakeManager(
        access_key_id="ak",
        secret_access_key="sk",
        region_name="region",
        endpoint_url="http://fake-s3",
    )
    data = b"hello"
    bucket = "bucket"
    key = "file.jpg"

    # Act
    mgr.upload_file(data, bucket, key)

    # Assert
    kw = _last_put_object_kwargs(mgr)
    assert kw["Bucket"] == bucket
    assert kw["Key"] == key
    assert kw["ContentType"] == "image/jpeg"
    assert kw["Body"] == data


def test_upload_image_resizes_and_saves_as_jpeg(fake_boto3):
    mgr = DataLakeManager("ak", "sk", "region", "http://fake-s3")

    # 임의의 BGR 이미지 (50x40, 파란색)
    img_bgr = np.zeros((40, 50, 3), dtype=np.uint8)
    img_bgr[..., 0] = 255  # B 채널

    # Act
    mgr.upload_image(img_bgr, "bucket", "thumb.jpg")

    # Assert
    kw = _last_put_object_kwargs(mgr)
    body = kw["Body"]
    with Image.open(io.BytesIO(body)) as im:
        assert im.format == "JPEG"
        assert im.size == (248, 140)  # 기본 썸네일 사이즈


def test_change_np2pil_jpg_bytes_custom_size():
    mgr = DataLakeManager("ak", "sk", "region", "http://fake-s3")

    # 1x1 BGR(255,0,0): 파란 픽셀 (B 채널=255)
    pixel_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)

    jpg_bytes = mgr.change_np2pil_jpg_bytes(pixel_bgr, thumbnail_size="1x1")

    with Image.open(io.BytesIO(jpg_bytes)) as im:
        r, g, b = im.convert("RGB").getpixel((0, 0))
        # JPEG 압축 오차 고려해서 근사치로 검사
        assert b > 200 and r < 50 and g < 50, f"Unexpected RGB pixel: {(r,g,b)}"
