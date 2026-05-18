from pia_prod.AI.modules.gangnam_falldown.service import InternVL3TrtFalldownService
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import VQABase
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from queue import Queue
import cv2
import os
import pytest


@pytest.fixture(scope="module")
def safety_video(nas_downloader, video_save_dir):

    video_name = "safety0001_clip.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)

    return local_video_path


def test_gangnam_internvl3(safety_video):
    cap = cv2.VideoCapture(safety_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            vqaEvent=[
                VQABase(
                    name="falldown_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    # roi=ROIModel(roiId=1, polygonCoordinates=[0, 0, 1900, 0, 1900, 2000, 0, 2000]),
                    roi=ROIModel(roiId=1, polygonCoordinates=[0, 0, 1000, 0, 300, 1000, 0, 1000]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    cnt = -1
    if cap.isOpened():
        InternVL3TrtFalldownService(q)
        while True:
            cnt += 1
            ret, frame = cap.read()
            if not ret:
                break

            if cnt % round(fps * interval) != 0:
                continue
            q.put(
                {
                    "batches": [frame, frame],
                    "stream_ids": [stream_id, stream_id + "2"],
                    "user_params": [user_param, user_param],
                }
            )

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import InternVL3TrtFalldownService

    assert InternVL3TrtFalldownService is not None, "InternVL3TrtFalldownService import failed"
