from pia_prod.AI.modules.yeonsei_smoke.service import SmokeService
from queue import Queue
from pia_prod.AI.modules.yeonsei_smoke.param import SmokeModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia.ai.exports.trt import onnx2trt
import cv2
import os
import pytest
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.modules.yeonsei_smoke.config import SMOKE_CLS_MODEL_ONNX_PATH, CLS_INPUT_SIZE


@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert SMOKE_CLS_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = os.getenv("MODEL_SMOKE_CLS_ONNX_PATH", "assets/model/SmokeCls_v1.2.0.onnx")
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
    video_name = "ys_smoke_video.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


def test_yeonsei_smoke(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            cvEvent=[
                SmokeModel(
                    name="smoke_cv",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    bbox_min_area=1000,
                    cls_threshold=0.8,
                    # lower_hsv=[0, 0, 0],
                    # upper_hsv=[0, 0, 0],
                    roi=ROIModel(
                        roiId=1,
                        # polygonCoordinates=[0, 0, 100, 0, 100, 100, 0, 100]
                    ),
                )
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    if cap.isOpened():
        SmokeService(q)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import SmokeService

    assert SmokeService is not None, "Import SmokeService Failed"
