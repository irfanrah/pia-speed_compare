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


def _build_user_param(camera_id: str, camera_url: str, organization: str, roi_coords):
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId=camera_id,
            cameraUrl=camera_url,
            organization=organization,
            vqaEvent=[
                VQABase(
                    name="falldown_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=roi_coords),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    return user_param


def test_gangnam_internvl3_roi_and_no_roi(safety_video):
    cap = cv2.VideoCapture(safety_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1
    q = Queue(1)

    user_param_roi = _build_user_param(
        camera_id="6",
        camera_url="11",
        organization="pia",
        roi_coords=[0, 0, 1000, 0, 300, 1000, 0, 1000],
    )
    user_param_no_roi = _build_user_param(
        camera_id="7",
        camera_url="12",
        organization="pia",
        roi_coords=[],
    )
    stream_id_roi = (
        f"{user_param_roi['user_param']['cameraId']}_{user_param_roi['user_param']['organization']}"
    )
    stream_id_no_roi = (
        f"{user_param_no_roi['user_param']['cameraId']}_{user_param_no_roi['user_param']['organization']}"
    )

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
                    "stream_ids": [stream_id_roi, stream_id_no_roi],
                    "user_params": [user_param_roi, user_param_no_roi],
                }
            )

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"
