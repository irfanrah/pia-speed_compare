from queue import Queue
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.modules.khonkaen_helmet.service import HelmetService
from pia_prod.AI.modules.khonkaen_helmet.param import HelmetModel
from pia_prod.AI.modules.khonkaen_helmet.config import (
    MODEL_BIKER_DETECTION_ONNX_PATH,
    MODEL_BIKER_DETECTION_TRT_PATH,
    MODEL_HELMET_CLS_TRT_PATH,
    MODEL_HELMET_CLS_ONNX_PATH,
    DEFAULT_FPS,
    OD_TIME_INTERVAL_SECOND,
    IMAGE_SAVE_PATH,
    OD_INPUT_SIZE,
    CLS_INPUT_SIZE,
)
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia.ai.exports.trt import onnx2trt
import cv2
import os
import time
import pytest
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.tests.conftest import set_logging_flag
from pia_prod.AI.modules.khonkaen_weapon.debug_utils import VideoSaveManager


@pytest.fixture(scope="module")
def get_od_model(hf_downloader):
    assert MODEL_BIKER_DETECTION_ONNX_PATH is not None, "Please, check model file path"
    model_path = MODEL_BIKER_DETECTION_ONNX_PATH
    # model download from huggingFace
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    # export model to TRT
    onnx2trt(
        onnx_path=model_path,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, OD_INPUT_SIZE[0], OD_INPUT_SIZE[1]),
        max_batch=16,
        opt_batch=8,
        min_batch=1,
    )

    assert os.path.exists(model_path), f"Model file {model_path} not found"
    assert os.path.exists(
        MODEL_BIKER_DETECTION_TRT_PATH
    ), f"TRT model file {MODEL_BIKER_DETECTION_TRT_PATH} not found"

    return model_path


@pytest.fixture(scope="module")
def get_cls_model(hf_downloader):
    # model download from huggingFace
    assert MODEL_HELMET_CLS_ONNX_PATH is not None, "Please, check model file path"
    model_path = MODEL_HELMET_CLS_ONNX_PATH
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
    # export model to TRT
    onnx2trt(
        onnx_path=model_path,
        device=0,
        fp16=True,
        overwrite=False,
        input_shape=(3, CLS_INPUT_SIZE[0], CLS_INPUT_SIZE[1]),
        max_batch=16 * 8,
        opt_batch=8 * 8,
        min_batch=1,
    )

    assert os.path.exists(model_path), f"Model file {model_path} not found"
    assert os.path.exists(
        MODEL_HELMET_CLS_TRT_PATH
    ), f"TRT model file {MODEL_HELMET_CLS_TRT_PATH} not found"

    return model_path


@pytest.fixture(scope="module")
def get_video_1(nas_downloader, video_save_dir):
    video_name = "kk_helmet_1.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)

    assert os.path.exists(
        local_video_path
    ), f"Video file {video_name} not found in {video_save_dir}"
    return local_video_path


@pytest.fixture(scope="module")
def get_video_2(nas_downloader, video_save_dir):
    video_name = "kk_helmet_2.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)

    assert os.path.exists(
        local_video_path
    ), f"Video file {video_name} not found in {video_save_dir}"
    return local_video_path


@pytest.fixture(scope="module")
def get_AddStreamModel() -> tuple:
    def _create_add_stream_model(camera_id: int) -> tuple:
        user_param = {}

        add_stream_instance = AddStreamModel(
            cameraId=str(camera_id),  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            cvEvent=[
                HelmetModel(
                    name="helmet_cv",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    confidenceThreshold=0.5,
                    nmsThreshold=0.5,
                    roi=ROIModel(
                        roiId=1, polygonCoordinates=[1912, 1070, 6, 1072, 11, 745, 1915, 353]
                    ),
                )
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
        user_param["user_param"] = AddStreamModel2dict(add_stream_instance)
        stream_id = (
            f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
        )
        assert isinstance(
            add_stream_instance, AddStreamModel
        ), "user_param should be an instance of AddStreamModel"
        return user_param, stream_id

    return _create_add_stream_model


def test_helmet_1batch(
    get_od_model,
    get_cls_model,
    get_video_2,
    get_AddStreamModel,
):
    cap_1 = cv2.VideoCapture(get_video_2)
    q = Queue(1)
    assert cap_1 is not None, "Failed to open video 1"

    count = 0
    video_speed = 8.0  # 비디오 재생 속도
    model_input_second = OD_TIME_INTERVAL_SECOND / video_speed  # 원래는 OD_TIME_INTERVAL_SECOND

    user_param, stream_id = get_AddStreamModel(1)

    count = 0
    if cap_1.isOpened():
        service = HelmetService(q)
        service.save_video = True
        service.video_manager = VideoSaveManager(
            output_dir=os.path.join(IMAGE_SAVE_PATH, stream_id, "helmet"),
            save_video=service.save_video,
            model_input_second=model_input_second,
            save_interval_seconds=10,
            prefix="video",
        )
        fps = cap_1.get(cv2.CAP_PROP_FPS)
        service.logging_flag = set_logging_flag()
        while True:
            ret, frame = cap_1.read()
            if not ret:
                break

            if count % round(fps * OD_TIME_INTERVAL_SECOND) == 0:
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})
            count += 1

        while q.qsize() > 0:
            time.sleep(1)

        service.video_manager.close()
        service.video_manager = None
        service.del_model()
        stop_thread(q)
    else:
        assert "Failed to open video 1", "Video capture could not be opened"


def test_helmet_batches(
    get_od_model, get_cls_model, get_video_1, get_video_2, get_AddStreamModel, batch_size=4
):
    cap_1 = cv2.VideoCapture(get_video_1)
    cap_2 = cv2.VideoCapture(get_video_2)
    q = Queue(1)
    assert cap_1 is not None, "Failed to open video 1"
    assert cap_2 is not None, "Failed to open video 2"

    user_params = []
    stream_ids = []
    for i in range(batch_size):
        user_param_i, stream_id_i = get_AddStreamModel(i)
        user_params.append(user_param_i)
        stream_ids.append(stream_id_i)

    count = 0
    if cap_1.isOpened() and cap_2.isOpened():
        service = HelmetService(q)
        service.logging_flag = set_logging_flag()

        while True:
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            if not ret_1 or not ret_2:
                break

            batches = [frame_1, frame_2, frame_2, frame_1]
            assert len(batches) == batch_size, f"Batch size should be {batch_size}"

            if count % round(DEFAULT_FPS * OD_TIME_INTERVAL_SECOND) == 0:
                q.put(
                    {
                        "batches": batches,
                        "stream_ids": stream_ids,
                        "user_params": user_params,
                    }
                )
            count += 1

        while q.qsize() > 0:
            time.sleep(1)

        service.del_model()
        stop_thread(q)
    else:
        assert False, "Video capture could not be opened"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import HelmetService

    assert HelmetService is not None, "HelmetService module import failed"
