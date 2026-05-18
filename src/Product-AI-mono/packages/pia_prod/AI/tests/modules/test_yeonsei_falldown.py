from pia_prod.AI.modules.yeonsei_falldown.service import FalldownService
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.modules.yeonsei_falldown.param import FalldownModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia.ai.exports.trt import onnx2trt
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from queue import Queue
import os
import cv2
import pytest
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.modules.yeonsei_falldown.config import (
    PERSON_KEYPOINT_MODEL_ONNX_PATH,
    KP_INPUT_SIZE,
)


@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert PERSON_KEYPOINT_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = PERSON_KEYPOINT_MODEL_ONNX_PATH
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    onnx2trt(
        onnx_path=model_path,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, KP_INPUT_SIZE[0], KP_INPUT_SIZE[1]),
        max_batch=16,
        opt_batch=8,
        min_batch=1,
    )


@pytest.fixture(scope="function")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "ys_falldown_video.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


def test_yeonsei_falldown(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            cvEvent=[
                FalldownModel(
                    name="falldown_cv",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    angle_threshold=40,
                    # bbox_keep_head_foot_ratio=0.5,
                    # bbox_cutting_margin_ratio=0.5,
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
        FalldownService(q)
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
    from pia_prod.AI import FalldownService

    assert FalldownService is not None, "Import FalldownService Failed"
