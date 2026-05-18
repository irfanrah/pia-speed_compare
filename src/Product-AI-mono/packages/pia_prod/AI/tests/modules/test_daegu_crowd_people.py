from pia_prod.AI.modules.DaeGu_crowd_people.service import CPService
from pia_prod.AI.modules.DaeGu_crowd_people.config import CROWD_PEOPLE_ONNX_MODEL_PATH
from pia_prod.AI.modules.DaeGu_crowd_people.param import CPBase
from queue import Queue
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
import pytest
import cv2
import os


@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert CROWD_PEOPLE_ONNX_MODEL_PATH is not None, "Please, check model file path"
    model_path = CROWD_PEOPLE_ONNX_MODEL_PATH
    model_path = os.getenv(
        "MODEL_CROWD_PEOPLE_ONNX_PATH", "assets/model/CLIP_EBC_nwpu_rmse_onnx.onnx"
    )
    file_name = os.path.basename(model_path)
    repo_id, ext = os.path.splitext(os.path.basename(model_path))
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)


@pytest.fixture(scope="function")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "DaeGu_crowd_people.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)  # assets/videos/video.mp4
    nas_video_path = nas_downloader.get_nas_path(
        video_name
    )  # http://172.168.47.36/Product_ai_mono/test_videos/DaeGu_crowd_people.mp4
    nas_downloader.download_file(nas_video_path, local_video_path)
    return local_video_path


def test_daegu_crowd_people(model_download, local_video_path):
    cap = cv2.VideoCapture(local_video_path)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            cvEvent=[
                CPBase(
                    name="crowded_cv",
                    people_threshold=30,
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[0, 0, 1900, 0, 1900, 2000, 0, 2000]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    cnt = 0
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    if cap.isOpened():
        CPService(q)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cnt += 1
            if cnt % 120 != 0:
                continue
            q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        stop_thread(q)

    else:
        assert False, "Can't Open the Video"
    print("Daegu Crowd People Test Finish")


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import CPService

    assert CPService is not None, "CPService import failed"
