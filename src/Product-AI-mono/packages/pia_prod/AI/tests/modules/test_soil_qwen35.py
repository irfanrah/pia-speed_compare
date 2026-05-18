import os
import pytest
import cv2
import httpx
from queue import Queue

from pia_prod.AI.modules.soil_qwen35.service import SoilQwen35Service
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import VQABase, ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread


@pytest.fixture(scope="module")
def check_vllm_server():
    """vLLM 서버 가용성 확인. 미실행 시 테스트 skip."""
    host = os.getenv("SOIL_QWEN35_VLLM_HOST", "localhost")
    port = os.getenv("SOIL_QWEN35_VLLM_PORT", "8000")
    url = f"http://{host}:{port}/v1"
    try:
        resp = httpx.get(f"{url}/models", timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip(
            "vLLM 서버가 실행 중이 아닙니다. 먼저 서버를 실행하세요:\n"
            "  cd packages/pia_prod/AI/modules/soil_qwen35/vllm_server\n"
            "  docker compose up -d"
        )


@pytest.fixture(scope="module")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "samsung_fire_fps13.mp4"
    local_path = os.path.join(video_save_dir, video_name)
    nas_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_path, local_path)
    return local_path


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import SoilQwen35Service

    assert SoilQwen35Service is not None, "SoilQwen35Service import failed"


def test_soil_qwen35_single(check_vllm_server, local_video_path):
    """단일 카메라 + 멀티 카테고리 테스트."""
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",
            cameraUrl="11",
            organization="pia",
            vqaEvent=[
                VQABase(
                    name="fire_qwen_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
                VQABase(
                    name="smoke_qwen_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    frame_interval = max(int(fps / 2), 1)
    cnt = 0
    if cap.isOpened():
        SoilQwen35Service(q)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cnt += 1
            if cnt % frame_interval == 1:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})
        stop_thread(q)
        cap.release()
    else:
        assert False, "Can't Open the Video"


def test_soil_qwen35_fire_only(check_vllm_server, local_video_path):
    """단일 카테고리만 등록하여 테스트."""
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",
            cameraUrl="11",
            organization="pia",
            vqaEvent=[
                VQABase(
                    name="fire_qwen_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    frame_interval = max(int(fps / 2), 1)
    cnt = 0
    if cap.isOpened():
        SoilQwen35Service(q)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cnt += 1
            if cnt % frame_interval == 1:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})
        stop_thread(q)
        cap.release()
    else:
        assert False, "Can't Open the Video"


def test_soil_qwen35_multiple_cameras(check_vllm_server, local_video_path):
    """멀티 카메라 배치 테스트."""
    cap_1 = cv2.VideoCapture(local_video_path)
    cap_2 = cv2.VideoCapture(local_video_path)
    q = Queue(1)
    user_param_1 = {}
    user_param_1["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",
            cameraUrl="11",
            organization="pia",
            vqaEvent=[
                VQABase(
                    name="fire_qwen_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    user_param_2 = {}
    user_param_2["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="7",
            cameraUrl="12",
            organization="pia",
            vqaEvent=[
                VQABase(
                    name="smoke_qwen_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
                VQABase(
                    name="falldown_qwen_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id_1 = f"{user_param_1['user_param']['cameraId']}_{user_param_1['user_param']['organization']}"
    stream_id_2 = f"{user_param_2['user_param']['cameraId']}_{user_param_2['user_param']['organization']}"

    fps = 15
    interval = 0.3
    count = 0
    if cap_1.isOpened() and cap_2.isOpened():
        SoilQwen35Service(q)
        while True:
            count += 1
            if count % round(fps * interval) == 0:
                ret_1, frame_1 = cap_1.read()
                ret_2, frame_2 = cap_2.read()
                if not ret_1 or not ret_2:
                    break
                q.put(
                    {
                        "batches": [frame_1, frame_2],
                        "stream_ids": [stream_id_1, stream_id_2],
                        "user_params": [user_param_1, user_param_2],
                    }
                )
        stop_thread(q)
        cap_1.release()
        cap_2.release()
    else:
        assert False, "Can't Open the Video"
