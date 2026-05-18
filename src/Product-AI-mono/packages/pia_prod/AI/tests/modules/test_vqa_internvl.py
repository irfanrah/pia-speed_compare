from pia_prod.AI.modules.vqa_internvl.service import InternVL3TrtLlmService
from pia_prod.AI.modules.vqa_internvl.config import HF_REPO_ID, ASSETS_MODEL_DIR, MODEL_DIR
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import VQABase
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from queue import Queue
import shutil
import cv2
import os
import pytest


@pytest.fixture(scope="module")
def vqa_video(nas_downloader, video_save_dir):
    video_name = "ys_falldown_video.mp4"
    local_path = os.path.join(video_save_dir, video_name)
    if not os.path.isfile(local_path):
        nas_video_path = nas_downloader.get_nas_path(video_name)
        nas_downloader.download_file(nas_video_path, local_path)
    return local_path


@pytest.fixture(scope="module")
def model_download(hf_downloader, vqa_video):
    # 1) assets/images/frame_0.jpg 준비 (VIT 엔진 빌드용 샘플 이미지)
    images_dir = "assets/images"
    frame_path = os.path.join(images_dir, "frame_0.jpg")
    if not os.path.isfile(frame_path):
        os.makedirs(images_dir, exist_ok=True)
        cap = cv2.VideoCapture(vqa_video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(frame_path, frame)

    # 2) 모델 다운로드 — MODEL_DIR에 config.json이 없으면 재다운로드
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        if os.path.isdir(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        hf_downloader.download(repo_id=HF_REPO_ID, save_dir=ASSETS_MODEL_DIR, snapshot=True)


def test_vqa_internvl_single(model_download, vqa_video):
    """영상 1채널, ROI 없음."""
    cap = cv2.VideoCapture(vqa_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 0.5  # 0.5초마다 1프레임
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="1",
            cameraUrl="rtsp://dummy",
            organization="pia",
            vqaEvent=[
                VQABase(
                    name="falldown_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    cnt = -1
    if cap.isOpened():
        InternVL3TrtLlmService(q)
        while True:
            cnt += 1
            ret, frame = cap.read()
            if not ret:
                break

            if cnt % max(1, round(fps * interval)) != 0:
                continue
            q.put(
                {
                    "batches": [frame],
                    "stream_ids": [stream_id],
                    "user_params": [user_param],
                }
            )

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_vqa_internvl_2ch(model_download, vqa_video):
    """동일 영상을 2채널(배치 2)로 동시 처리, ROI 없음."""
    cap = cv2.VideoCapture(vqa_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 0.5  # 0.5초마다 1프레임
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="2",
            cameraUrl="rtsp://dummy",
            organization="pia",
            vqaEvent=[
                VQABase(
                    name="falldown_vqa",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    cnt = -1
    if cap.isOpened():
        InternVL3TrtLlmService(q)
        while True:
            cnt += 1
            ret, frame = cap.read()
            if not ret:
                break

            if cnt % max(1, round(fps * interval)) != 0:
                continue
            q.put(
                {
                    "batches": [frame, frame],
                    "stream_ids": [stream_id, stream_id + "_2"],
                    "user_params": [user_param, user_param],
                }
            )

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import InternVL3TrtLlmService

    assert InternVL3TrtLlmService is not None, "InternVL3TrtLlmService import failed"
