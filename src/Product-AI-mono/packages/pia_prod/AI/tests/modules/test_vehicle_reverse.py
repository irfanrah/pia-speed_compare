from queue import Queue
from typing import Tuple
import os
import time
import cv2
import pytest

from pia.ai.exports.trt import onnx2trt
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.modules.vehicle_reverse.param import VehicleReverseModel
from pia_prod.AI.modules.vehicle_reverse.service import VehicleReverseService
from pia_prod.AI.modules.vehicle_reverse.config import (
    MODEL_VEHICLE_REVERSE_ONNX_PATH,
    MODEL_VEHICLE_REVERSE_TRT_PATH,
    REID_MODEL_PATH,
    IMAGE_SAVE_PATH,
)
from pia_prod.AI.modules.vehicle_reverse.debug_utils import save_snapshot_for_vehicle_reverse
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.conftest import set_logging_flag
from pia_prod.AI.modules.khonkaen_weapon.debug_utils import VideoSaveManager

@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert MODEL_VEHICLE_REVERSE_ONNX_PATH is not None, "Please, check model file path"
    model_path = MODEL_VEHICLE_REVERSE_ONNX_PATH
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    onnx2trt(
        onnx_path=model_path,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, 640, 640),
        max_batch=16,
        opt_batch=8,
        min_batch=1,
        engine_path=MODEL_VEHICLE_REVERSE_TRT_PATH,
    )
    assert REID_MODEL_PATH is not None, "Please, check reid model file path"

    reid_file_name = os.path.basename(REID_MODEL_PATH)
    reid_repo_id, _ = os.path.splitext(reid_file_name)
    hf_downloader.download(
        repo_id=reid_repo_id,
        file_name=reid_file_name,
        snapshot=False,
    )


@pytest.fixture(scope="function")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "vehicle_reverse_2.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


def _build_user_param(camera_id: str, organization: str = "pia") -> Tuple[dict, str]:
    # Map arbitrary camera ids to deterministic integers to satisfy AddStreamModel validation.
    if not hasattr(_build_user_param, "_id_map"):
        _build_user_param._id_map = {}
        _build_user_param._next_id = 600
    if isinstance(camera_id, int):
        numeric_camera_id = camera_id
    else:
        if camera_id not in _build_user_param._id_map:
            _build_user_param._id_map[camera_id] = _build_user_param._next_id
            _build_user_param._next_id += 1
        numeric_camera_id = _build_user_param._id_map[camera_id]

    add_stream = AddStreamModel(
        cameraId=numeric_camera_id,
        cameraUrl="dummy_url",
        organization=organization,
        cvEvent=[
            VehicleReverseModel(
                name="vehiclereverse_cv",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                confidenceThreshold=0.5,
                nmsThreshold=0.5,
                roi=ROIModel(
                    roiId=1,
                    polygonCoordinates=[20, 1074, 1357, 1075, 1608, 374, 1541, 368, 6, 666],
                    divideCoordinates=[200, 1077, 1584, 372],
                ),
            )
        ],
        timestamp=str_UTC_ISO8601_ms_now_time(),
    )
    user_param = {"user_param": AddStreamModel2dict(add_stream)}
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    return user_param, stream_id


def test_vehicle_reverse_single(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    assert cap.isOpened(), "Can't open video"

    q = Queue(1)

    user_param, stream_id = _build_user_param("vehicle_reverse_cam")
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    time_interval = 0.125
    
    video_speed = 4.0  # 비디오 재생 속도
    model_input_second = time_interval / video_speed  # 원래는 OD_TIME_INTERVAL_SECOND
    count = 0
    service = VehicleReverseService(q)
    service.logging_flag = set_logging_flag()
    service.save_video = True
    service.video_manager = VideoSaveManager(
        output_dir=os.path.join(IMAGE_SAVE_PATH, stream_id, "vehicle_reverse"),
        save_video=service.save_video,
        model_input_second=model_input_second,
        save_interval_seconds=10,
        prefix="video",
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(fps * time_interval) == 0:
            q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})
        count += 1

    while q.qsize() > 0:
        time.sleep(1)

    service.video_manager.close()
    service.video_manager = None
    stop_thread(q)
    cap.release()


def test_vehicle_reverse_multi(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    assert cap.isOpened(), "Can't open video"

    q = Queue(1)

    user_params = []
    stream_ids = []
    base_camera_id = "vehicle_reverse_cam"
    organization = "pia"

    for i in range(8):
        user_param, stream_id = _build_user_param(
            f"{base_camera_id}_{i}", organization=organization
        )
        user_params.append(user_param)
        stream_ids.append(stream_id)

    count = 0
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    time_interval = 0.25

    service = VehicleReverseService(q)
    service.logging_flag = set_logging_flag(True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(fps * time_interval) == 0:
            q.put({"batches": [frame] * 8, "stream_ids": stream_ids, "user_params": user_params})
        count += 1

    while q.qsize() > 0:
        time.sleep(1)

    stop_thread(q)
    cap.release()


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import VehicleReverseService

    assert VehicleReverseService is not None, "VehicleReverseService module import failed"
