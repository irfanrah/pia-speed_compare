from queue import Queue
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.modules.ax_harness.service import HarnessService
from pia_prod.AI.modules.ax_harness.config import HARNESS_DETECTION_MODEL_ONNX_PATH, OD_INPUT_SIZE
from pia_prod.AI.modules.ax_harness.param import HarnessModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia.ai.exports.trt import onnx2trt
from pia_prod.AI.tests.thread_stop_signal import stop_thread
import cv2
import os
import pytest


@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert HARNESS_DETECTION_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = HARNESS_DETECTION_MODEL_ONNX_PATH
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    onnx2trt(
        onnx_path=HARNESS_DETECTION_MODEL_ONNX_PATH,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, OD_INPUT_SIZE[0], OD_INPUT_SIZE[1]),
        max_batch=16,
        opt_batch=8,
        min_batch=1,
    )


@pytest.fixture(scope="function")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "ax_harness_2.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


def test_ax_harness(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 0.3
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            cvEvent=[
                HarnessModel(
                    name="harnessoff_cv",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    confidenceThreshold=0.5,
                    nmsThreshold=0.5,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                )
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    count = 0
    if cap.isOpened():
        HarnessService(q)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % round(fps * interval) == 0:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})
            count += 1
        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_ax_harness_8batch(model_download, local_video_path):
    """
    단일 영상의 동일 프레임을 8개의 서로 다른 스트림(=배치 8)으로 전달하여
    HarnessService의 멀티 배치 처리를 검증한다.
    """
    cap = cv2.VideoCapture(local_video_path)
    assert cap is not None, "VideoCapture 생성 실패"
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 프레임 전송 간격 (초)
    interval = 0.3
    step = max(1, round(fps * interval))  # 0 방지

    q = Queue(1)

    # 8개 배치용 user_params / stream_ids 생성
    user_params = []
    stream_ids = []
    base_camera_id = "6"  # 기존 코드와 동일한 베이스
    organization = "pia"

    for i in range(8):
        up = {}
        up["user_param"] = AddStreamModel2dict(
            AddStreamModel(
                cameraId=f"{base_camera_id}_{i}",  # 각 배치마다 다른 cameraId
                cameraUrl=f"dummy_url_{i}",  # 더미
                organization=organization,  # 더미
                cvEvent=[
                    HarnessModel(
                        name="harnessoff_cv",
                        incidentThresholdSecond=3,
                        incidentTimeoutSecond=3,
                        confidenceThreshold=0.5,
                        nmsThreshold=0.5,
                        roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    )
                ],
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
        user_params.append(up)
        stream_ids.append(f"{up['user_param']['cameraId']}_{organization}")

    count = 0
    if cap.isOpened():
        HarnessService(q)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % step == 0:
                    # 동일 프레임을 8개 배치로 전송
                    q.put(
                        {
                            "batches": [frame] * 8,
                            "stream_ids": stream_ids,
                            "user_params": user_params,
                        }
                    )
                count += 1
        finally:
            stop_thread(q)
            cap.release()
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """모듈 임포트 테스트"""
    from pia_prod.AI import HarnessService

    assert HarnessService is not None, "HarnessService import failed"
