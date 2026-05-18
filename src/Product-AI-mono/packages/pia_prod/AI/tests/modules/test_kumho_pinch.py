import cv2
import os
import pytest

from queue import Queue, Empty

from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.modules.kumho_pinch.param import PinchModel
from pia_prod.AI.modules.kumho_pinch.service import PinchService
from pia_prod.AI.modules.kumho_pinch.config import (
    PERSON_DETECTION_MODEL_ONNX_PATH,
    OD_INPUT_SIZE,
)
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia.ai.exports.trt import onnx2trt
from pia_prod.AI.tests.thread_stop_signal import stop_thread


@pytest.fixture(scope="module")
def model_download(hf_downloader):
    assert PERSON_DETECTION_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = PERSON_DETECTION_MODEL_ONNX_PATH
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    onnx2trt(
        onnx_path=model_path,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, OD_INPUT_SIZE[0], OD_INPUT_SIZE[1]),
        max_batch=16,
        opt_batch=8,
        min_batch=1,
    )


# 임시 더미 데이터 pinch_1.mp4 파일로 교체
@pytest.fixture(scope="module")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "pinch_1.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)

    nas_video_path = nas_downloader.get_nas_path(video_name)

    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


@pytest.fixture(scope="module")
def get_multiple_videos(nas_downloader, video_save_dir):
    video_names = [
        "pinch_1.mp4",
        "pinch_1.mp4",
        "pinch_1.mp4",
        "pinch_1.mp4",
    ]

    local_video_paths = []
    for video_name in video_names:
        local_video_path = os.path.join(video_save_dir, video_name)
        nas_video_path = nas_downloader.get_nas_path(video_name)
        nas_downloader.download_file(nas_video_path, local_video_path)
        local_video_paths.append(local_video_path)

    return local_video_paths


def _build_user_param_dict(camera_id: str, organization: str, roi_id: int):
    """AddStreamModel과 AddStreamModel2dict을 사용하여 user_param 생성"""
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId=camera_id,
            cameraUrl="11",  # 더미값
            organization=organization,
            cvEvent=[
                PinchModel(
                    name="pinch_cv",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    confidenceThreshold=0.5,
                    nmsThreshold=0.5,
                    roi=ROIModel(
                        roiId=roi_id,
                        polygonCoordinates=[714, 0, 875, 1072, 0, 1073, 0, 0],
                        divideCoordinates=[],
                    ),
                )
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    return user_param


def _try_get_result_queue(service):
    """Service 객체에서 result queue를 찾아 반환"""
    for attr in ("result_queue", "out_queue", "output_queue", "result_q", "out_q", "output_q"):
        q = getattr(service, attr, None)
        if isinstance(q, Queue):
            return q
    return None


def test_single_camera(model_download, local_video_path):
    """단일 카메라 테스트"""
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 0.3
    q = Queue(1)

    user_param = _build_user_param_dict(camera_id="6", organization="pia", roi_id=1)

    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    count = 0

    if cap.isOpened():
        pinch_service = PinchService(q)
        result_q = _try_get_result_queue(pinch_service)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % round(fps * interval) == 0:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

                if result_q is not None:
                    try:
                        result = result_q.get_nowait()
                    except Empty:
                        result = None

                    if isinstance(result, dict):
                        detected_person = bool(
                            result.get("detections")
                            or result.get("persons")
                            or result.get("person")
                        )
                        triggered_event = bool(
                            result.get("events") or result.get("event") or result.get("incidents")
                        )
                        print(
                            f"[{stream_id}] person_detected={detected_person}, "
                            f"event_triggered={triggered_event}"
                        )

            count += 1

        stop_thread(q)
        cap.release()
    else:
        assert False, "Can't Open the Video"


def test_multiple_cameras(model_download, get_multiple_videos, batch_size=4):
    """다중 카메라 배치 테스트"""
    video_paths = get_multiple_videos
    assert len(video_paths) == 4, "4개의 비디오 경로가 필요합니다"

    cap_1 = cv2.VideoCapture(video_paths[0])
    cap_2 = cv2.VideoCapture(video_paths[1])
    cap_3 = cv2.VideoCapture(video_paths[2])
    cap_4 = cv2.VideoCapture(video_paths[3])

    fps = cap_1.get(cv2.CAP_PROP_FPS)
    interval = 0.3
    q = Queue(1)

    user_params = []
    stream_ids = []

    # 각 카메라에 대한 user_param 생성
    for i in range(batch_size):
        user_param = _build_user_param_dict(camera_id=str(i), organization="pia", roi_id=i)
        stream_id = (
            f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
        )
        user_params.append(user_param)
        stream_ids.append(stream_id)

    count = 0

    if cap_1.isOpened() and cap_2.isOpened() and cap_3.isOpened() and cap_4.isOpened():
        pinch_service = PinchService(q)
        result_q = _try_get_result_queue(pinch_service)

        while True:
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            ret_3, frame_3 = cap_3.read()
            ret_4, frame_4 = cap_4.read()

            if not ret_1 or not ret_2 or not ret_3 or not ret_4:
                break

            batches = [frame_1, frame_2, frame_3, frame_4]

            if count % round(fps * interval) == 0:
                q.put({"batches": batches, "stream_ids": stream_ids, "user_params": user_params})

                if result_q is not None:
                    try:
                        result = result_q.get_nowait()
                    except Empty:
                        result = None

                    if isinstance(result, dict):
                        detections = result.get("detections") or result.get("persons") or []
                        events = result.get("events") or result.get("incidents") or []
                        detected_any_person = bool(detections)
                        triggered_any_event = bool(events)
                        print(
                            f"[multi] person_detected={detected_any_person}, "
                            f"event_triggered={triggered_any_event}"
                        )

            count += 1

        stop_thread(q)
        cap_1.release()
        cap_2.release()
        cap_3.release()
        cap_4.release()
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """모듈 import 테스트"""
    from pia_prod.AI import PinchService

    assert PinchService is not None, "PinchService module import failed"
