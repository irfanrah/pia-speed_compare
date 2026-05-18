import os
import cv2
from queue import Queue

from pia_prod.AI.modules.samsung_fire.service import FireService
from pia_prod.AI.modules.samsung_fire.config import FIRE_CLS_MODEL_ONNX_PATH, CLS_INPUT_SIZE
from pia_prod.AI.modules.samsung_fire.param import FireModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia.ai.exports.trt import onnx2trt
from pia_prod.AI.tests.thread_stop_signal import stop_thread
import pytest


@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert FIRE_CLS_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = FIRE_CLS_MODEL_ONNX_PATH
    model_path = os.getenv(
        "MODEL_FIRE_CLS_ONNX_PATH", "assets/model/FireMotionColorYOLO_v0.1.0.onnx"
    )
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    onnx2trt(
        onnx_path=model_path,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, CLS_INPUT_SIZE[0], CLS_INPUT_SIZE[1]),
        max_batch=16,
        opt_batch=8,
        min_batch=1,
    )


@pytest.fixture(scope="function")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "samsung_fire.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


def test_samsung_fire(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    q = Queue(1)

    user_param = {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId="7",
                cameraUrl="rtsp://dummy",
                organization="pia",
                cvEvent=[
                    FireModel(
                        name="fire_cv",
                        incidentThresholdSecond=3,
                        incidentTimeoutSecond=3,
                        bbox_min_area=1000,
                        cls_threshold=0.8,
                        roi=ROIModel(
                            roiId=1,
                            # polygonCoordinates=[0, 0, 100, 0, 100, 100, 0, 100]
                        ),
                    )
                ],
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }

    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    if cap.isOpened():
        FireService(q, save_csv=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        stop_thread(q)
    else:
        assert False, "Cannot open the video file"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import FireService

    assert FireService is not None, "FireService module import failed"
