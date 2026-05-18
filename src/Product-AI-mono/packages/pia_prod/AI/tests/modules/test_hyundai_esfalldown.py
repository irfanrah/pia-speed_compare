from pia_prod.AI.modules.hyundai_esfalldown.service import ESFalldownService as PEService
from pia_prod.AI.modules.hyundai_esfalldown.config import (
    PERCEPTION_ENCODER_TXT_FEATURE_PATH,
    PERCEPTION_ENCODER_TRT_PATH,
    PERCEPTION_ENCODER_ONNX_PATH,
    PERCEPTION_ENCODER_PYTORCH_PATH,
    INPUT_SIZE,
)
from queue import Queue
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import RetrievalBase
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.modules.perception_encoder.trt_export import export_trt_engine
import threading
import torch
import pytest
import cv2
import os


@pytest.fixture
def setup_model_and_video(hf_downloader, nas_downloader, video_save_dir):
    txt_json_name = os.path.basename(PERCEPTION_ENCODER_TXT_FEATURE_PATH)
    onnx_file_save_path = PERCEPTION_ENCODER_ONNX_PATH
    repo_name = os.path.splitext(os.path.basename(PERCEPTION_ENCODER_PYTORCH_PATH))[0]
    onnx_repo_name = os.path.splitext(os.path.basename(onnx_file_save_path))[0]
    download_model_name = os.path.basename(onnx_file_save_path)

    # download .onnx file
    hf_downloader.download(repo_id=onnx_repo_name, file_name=download_model_name, snapshot=False)
    # download .json file
    hf_downloader.download(repo_id=repo_name, file_name=txt_json_name, snapshot=False)
    export_trt_engine(
        onnx_file=onnx_file_save_path,
        save_file_path=PERCEPTION_ENCODER_TRT_PATH,
        min_batch_size=1,
        opt_batch_size=4,
        max_batch_size=8,
        input_size=INPUT_SIZE,
    )

    video_name_1 = "20230518_sinmae_esfalldown.mp4"
    video_name_2 = "20230609_hwawon_esfalldown.mp4"
    video_name_3 = "ec_10_15-16_downstation_bottom_up_falldown_2_esfalldown.mp4"
    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)
    local_video_path_3 = os.path.join(video_save_dir, video_name_3)

    nas_path_1 = nas_downloader.get_nas_path(video_name_1)
    nas_path_2 = nas_downloader.get_nas_path(video_name_2)
    nas_path_3 = nas_downloader.get_nas_path(video_name_3)
    nas_downloader.download_file(nas_path_1, local_video_path_1)
    nas_downloader.download_file(nas_path_2, local_video_path_2)
    nas_downloader.download_file(nas_path_3, local_video_path_3)

    return local_video_path_1, local_video_path_2, local_video_path_3


def test_perception_encoder(setup_model_and_video):
    # model download from huggingFace
    # TODO : 현재는 pia-ai-package에 있는 모델 경로로만 실행 가능
    # assert os.getenv("PERCEPTION_ENCODER_PYTORCH_PATH") != "", "Please, check model file path"

    video_path_1, video_path_2, video_path_3 = setup_model_and_video
    cap = cv2.VideoCapture(video_path_1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="esfalldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    frame_interval = int(fps / 2)
    cnt = 0
    if cap.isOpened():
        service = PEService(q)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            cnt += 1
            if cnt % frame_interval == 1:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        service.thread_state = False
        for th in threading.enumerate():
            if th.name == "object_detection_ai_inference":
                th.join()
                print("완료")
    else:
        assert False, "Can't Open the Video"


def test_perception_encoder_multiple_cameras(setup_model_and_video):
    video_path_1, video_path_2, video_path_3 = setup_model_and_video
    cap_1 = cv2.VideoCapture(video_path_1)
    cap_2 = cv2.VideoCapture(video_path_2)
    cap_3 = cv2.VideoCapture(video_path_3)
    q = Queue(1)
    user_param_1 = {}
    user_param_1["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="esfalldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(
                        roiId=1,
                        polygonCoordinates=[],
                    ),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    user_param_2 = {}
    user_param_2["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="7",  # 더미값 넣어줌
            cameraUrl="12",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="에스컬레이터쓰러짐_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    user_param_3 = {}
    user_param_3["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="8",  # 더미값 넣어줌
            cameraUrl="13",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="에스컬레이터쓰러짐_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id_1 = (
        f"{user_param_1['user_param']['cameraId']}_{user_param_1['user_param']['organization']}"
    )
    stream_id_2 = (
        f"{user_param_2['user_param']['cameraId']}_{user_param_2['user_param']['organization']}"
    )
    stream_id_3 = (
        f"{user_param_3['user_param']['cameraId']}_{user_param_3['user_param']['organization']}"
    )

    fps = 15
    interval = 0.3
    count = 0
    if cap_1.isOpened() and cap_2.isOpened() and cap_3.isOpened():
        PEService(q)
        while True:
            if count % round(fps * interval) == 0:
                ret_1, frame_1 = cap_1.read()
                ret_2, frame_2 = cap_2.read()
                ret_3, frame_3 = cap_3.read()
                if not ret_1 or not ret_2 or not ret_3:
                    break
                q.put(
                    {
                        "batches": [frame_1, frame_2, frame_3],
                        "stream_ids": [stream_id_1, stream_id_2, stream_id_3],
                        "user_params": [user_param_1, user_param_2, user_param_3],
                    }
                )
            count += 1

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_esfalldown_torch_one_camera(setup_model_and_video):
    video_path_1, _, _ = setup_model_and_video
    cap = cv2.VideoCapture(video_path_1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)
    user_param = {}
    user_param["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="esfalldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    frame_interval = int(fps / 2)
    cnt = 0
    if cap.isOpened():
        service = PEService(q)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            torched_frame = torch.from_numpy(frame)
            torched_frame = torched_frame[..., [2, 1, 0]]  # BGR to RGB

            cnt += 1
            if cnt % frame_interval == 1:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})


        stop_thread(q)
    else:
        assert False, "Can't Open the Video"

def test_esfalldown_torch_two_cameras(setup_model_and_video):
    video_path_1, video_path_2, _ = setup_model_and_video
    cap_1 = cv2.VideoCapture(video_path_1)
    cap_2 = cv2.VideoCapture(video_path_2)
    q = Queue(1)
    user_param_1 = {}
    user_param_1["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="6",  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="esfalldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    user_param_2 = {}
    user_param_2["user_param"] = AddStreamModel2dict(
        AddStreamModel(
            cameraId="7",  # 더미값 넣어줌
            cameraUrl="12",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            retEvent=[
                RetrievalBase(
                    name="에스컬레이터쓰러짐_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["esfalldown"],
                    normalText=["normal"],
                ),
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
    )
    stream_id_1 = (
        f"{user_param_1['user_param']['cameraId']}_{user_param_1['user_param']['organization']}"
    )
    stream_id_2 = (
        f"{user_param_2['user_param']['cameraId']}_{user_param_2['user_param']['organization']}"
    )
    
    fps = 15
    interval = 0.3
    count = 0
    if cap_1.isOpened() and cap_2.isOpened():
        PEService(q)
        while True:
            if count % round(fps * interval) == 0:
                ret_1, frame_1 = cap_1.read()
                ret_2, frame_2 = cap_2.read()
                if not ret_1 or not ret_2:
                    break
                torched_frame_1 = torch.from_numpy(frame_1)
                torched_frame_1 = torched_frame_1[..., [2, 1, 0]]  # BGR -> RGB
                torched_frame_2 = torch.from_numpy(frame_2)
                torched_frame_2 = torched_frame_2[..., [2, 1, 0]]  # BGR -> RGB
                q.put(
                    {
                        "batches": [torched_frame_1, torched_frame_2],
                        "stream_ids": [stream_id_1, stream_id_2],
                        "user_params": [user_param_1, user_param_2],
                    }
                )
            count += 1

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import PEService

    assert PEService is not None, "PEService import failed"
