from pia_prod.AI.modules.pe_distribution.service import PEDistributionService
from pia_prod.AI.modules.pe_distribution.config import (
    PE_DISTRIBUTION_TXT_FEATURE_PATH,
    PE_DISTRIBUTION_TRT_PATH,
    PE_DISTRIBUTION_ONNX_PATH,
    PE_DISTRIBUTION_PYTORCH_PATH,
    INPUT_SIZE,
)
from queue import Queue
import torch
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import RetrievalBase
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.modules.pe_distribution.trt_export import export_trt_engine
import pytest
import cv2
import os


@pytest.fixture
def setup_model_and_video(hf_downloader, nas_downloader, video_save_dir):
    txt_json_name = os.path.basename(PE_DISTRIBUTION_TXT_FEATURE_PATH)
    onnx_file_save_path = PE_DISTRIBUTION_ONNX_PATH
    repo_name = os.path.splitext(os.path.basename(PE_DISTRIBUTION_PYTORCH_PATH))[0]
    onnx_repo_name = os.path.splitext(os.path.basename(onnx_file_save_path))[0]
    download_model_name = os.path.basename(onnx_file_save_path)

    # download .onnx file
    hf_downloader.download(repo_id=onnx_repo_name, file_name=download_model_name, snapshot=False)
    # download .json file
    hf_downloader.download(repo_id=repo_name, file_name=txt_json_name, snapshot=False)
    export_trt_engine(
        onnx_file=onnx_file_save_path,
        save_file_path=PE_DISTRIBUTION_TRT_PATH,
        min_batch_size=1,
        opt_batch_size=4,
        max_batch_size=8,
        input_size=INPUT_SIZE,
    )

    video_name_1 = "ys_falldown_video.mp4"
    video_name_2 = "samsung_fire_fps13.mp4"
    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)

    nas_path_1 = nas_downloader.get_nas_path(video_name_1)
    nas_path_2 = nas_downloader.get_nas_path(video_name_2)
    nas_downloader.download_file(nas_path_1, local_video_path_1)
    nas_downloader.download_file(nas_path_2, local_video_path_2)

    return local_video_path_1, local_video_path_2


def test_pe_distribution(setup_model_and_video):
    video_path_1, video_path_2 = setup_model_and_video
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
                    name="smoke_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["smoke"],
                    normalText=["normal"],
                ),
                RetrievalBase(
                    name="falldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(
                        roiId=1,
                        # polygonCoordinates=[],  # ROI 없음, 감지되어야함
                        # polygonCoordinates=[1646,1402,2582,1404,2589,390,1570,376],  # 제대로된 ROI
                        polygonCoordinates=[
                            0,
                            1871,
                            1339,
                            1905,
                            1346,
                            358,
                            83,
                            355,
                        ],  # 잘못된 ROI, 대상이 ROI 내에 없음
                    ),
                    topCandidates=5,
                    abnormalText=["falldown"],
                    normalText=["normal"],
                ),
                RetrievalBase(
                    name="fire_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["fire"],
                    normalText=["normal"],
                ),
                RetrievalBase(
                    name="흡연_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["smoking"],
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
        PEDistributionService(q)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            cnt += 1
            if cnt % frame_interval == 1:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_pe_distribution_without_falldown(setup_model_and_video):
    video_path_1, video_path_2 = setup_model_and_video
    cap = cv2.VideoCapture(video_path_2)
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
                    name="fire_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["fire"],
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
        PEDistributionService(q)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            cnt += 1
            if cnt % frame_interval == 1:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_pe_distribution_multiple_cameras(setup_model_and_video):
    video_path_1, video_path_2 = setup_model_and_video
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
                    name="smoke_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(
                        roiId=1,
                        polygonCoordinates=[],
                    ),
                    topCandidates=5,
                    abnormalText=["smoke"],
                    normalText=["normal"],
                ),
                RetrievalBase(
                    name="falldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(
                        roiId=1, polygonCoordinates=[1646, 1402, 2582, 1404, 2589, 390, 1570, 376]
                    ),
                    topCandidates=5,
                    abnormalText=["falldown"],
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
                    name="fire_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(
                        roiId=1,
                        polygonCoordinates=[
                            1646,
                            1402,
                            2582,
                            1404,
                            2589,
                            390,
                            1570,
                            376,
                        ],  # 안먹혀야 정상
                    ),
                    topCandidates=5,
                    abnormalText=["fire"],
                    normalText=["normal"],
                ),
                RetrievalBase(
                    name="falldown_ret",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    roi=ROIModel(roiId=1, polygonCoordinates=[]),
                    topCandidates=5,
                    abnormalText=["falldown"],
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
        PEDistributionService(q)
        while True:
            if count % round(fps * interval) == 0:
                ret_1, frame_1 = cap_1.read()
                ret_2, frame_2 = cap_2.read()
                if not ret_1 or not ret_2:
                    break
                q.put(
                    {
                        "batches": [frame_1, frame_2],
                        "stream_ids": [stream_id_1, stream_id_2],
                        "user_params": [user_param_1, user_param_2],
                    }
                )

        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import PEDistributionService

    assert PEDistributionService is not None, "PEDistributionService import failed"
