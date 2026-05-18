from queue import Queue
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.yonsei_walljump.service import WalljumpService
from pia_prod.AI.modules.yonsei_walljump.param import WalljumpModel
from pia_prod.AI.modules.yonsei_walljump.config import (
    PERSON_DETECTION_MODEL_ONNX_PATH,
    PERSON_DETECTION_MODEL_TRT_PATH,
    MODEL_WALLJUMP_CLS_TRT_PATH,
    MODEL_WALLJUMP_CLS_ONNX_PATH,
    OD_TIME_INTERVAL_SECOND,
    OD_INPUT_SIZE,
    CLS_INPUT_SIZE,
)
from pia_prod.AI.global_config import (
    USER_PARAM_KEY,
    CV_EVENT_KEY,
    OD_THRESHOLD_KEY,
    CLS_THRESHOLD_KEY,
    IOU_THRESHOLD_KEY,
    CAMERA_ID_KEY,
    ORGANIZATION_KEY,
)

from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia.ai.exports.trt import onnx2trt
import cv2
import os
import pytest
from pia_prod.AI.tests.thread_stop_signal import stop_thread

OD_TEST_THRESHOLD = 0.444
IOU_TEST_THRESHOLD = 0.555
CLS_TEST_THRESHOLD = 0.666


@pytest.fixture(scope="module")
def get_od_model(hf_downloader):
    # model download from huggingFace
    assert PERSON_DETECTION_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = PERSON_DETECTION_MODEL_ONNX_PATH
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
        PERSON_DETECTION_MODEL_TRT_PATH
    ), f"TRT model file {PERSON_DETECTION_MODEL_TRT_PATH} not found"

    return model_path


@pytest.fixture(scope="module")
def get_cls_model(hf_downloader):
    # model download from huggingFace
    assert MODEL_WALLJUMP_CLS_ONNX_PATH is not None, "Please, check model file path"
    model_path = MODEL_WALLJUMP_CLS_ONNX_PATH
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
        MODEL_WALLJUMP_CLS_TRT_PATH
    ), f"TRT model file {MODEL_WALLJUMP_CLS_TRT_PATH} not found"

    return model_path


@pytest.fixture(scope="module")
def get_video_1(nas_downloader, video_save_dir):
    video_name = "ys_walljump_1.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)

    assert os.path.exists(
        local_video_path
    ), f"Video file {video_name} not found in {video_save_dir}"
    return local_video_path


@pytest.fixture(scope="module")
def get_video_2(nas_downloader, video_save_dir):
    video_name = "ys_walljump_2.mp4"
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

        WALLJUMP_CV_KEY = "walljump_cv"
        user_param = {}

        add_stream_instance = AddStreamModel(
            cameraId=str(camera_id),  # 더미값 넣어줌
            cameraUrl="11",  # 더미값 넣어줌
            organization="pia",  # 더미값 넣어줌
            cvEvent=[
                WalljumpModel(
                    name=WALLJUMP_CV_KEY,
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    od_threshold=OD_TEST_THRESHOLD,
                    iou_threshold=IOU_TEST_THRESHOLD,
                    cls_threshold=CLS_TEST_THRESHOLD,
                    roi=ROIModel(
                        roiId=1, polygonCoordinates=[571, 948, 489, 275, 1982, 327, 1918, 985]
                    ),
                )
            ],
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
        user_param[USER_PARAM_KEY] = AddStreamModel2dict(add_stream_instance)
        stream_id = ServiceBase.make_inference_key(
            user_param[USER_PARAM_KEY][CAMERA_ID_KEY],
            user_param[USER_PARAM_KEY][ORGANIZATION_KEY],
        )
        assert WALLJUMP_CV_KEY in user_param[USER_PARAM_KEY][CV_EVENT_KEY]
        cv_event = user_param[USER_PARAM_KEY][CV_EVENT_KEY]
        assert OD_TEST_THRESHOLD == cv_event[WALLJUMP_CV_KEY][OD_THRESHOLD_KEY]
        assert IOU_TEST_THRESHOLD == cv_event[WALLJUMP_CV_KEY][IOU_THRESHOLD_KEY]
        assert CLS_TEST_THRESHOLD == cv_event[WALLJUMP_CV_KEY][CLS_THRESHOLD_KEY]

        assert isinstance(
            add_stream_instance, AddStreamModel
        ), "user_param should be an instance of AddStreamModel"

        return user_param, stream_id

    return _create_add_stream_model


def test_walljump_1batch(
    get_od_model,
    get_cls_model,
    get_video_1,
    get_AddStreamModel,
):
    cap_1 = cv2.VideoCapture(get_video_1)
    q = Queue(1)
    assert cap_1 is not None, "Failed to open video 1"

    user_param, stream_id = get_AddStreamModel(1)
    fps = cap_1.get(cv2.CAP_PROP_FPS)
    count = 0
    if cap_1.isOpened():
        WalljumpService(q)
        while True:
            ret, frame = cap_1.read()
            if not ret:
                break

            if count % round(fps * OD_TIME_INTERVAL_SECOND) == 0:
                # cv2.imwrite("0.jpg", frame)
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})
                # time.sleep(OD_TIME_INTERVAL_SECOND - 0.01)
            count += 1

        stop_thread(q)
    else:
        assert "Failed to open video 1", "Video capture could not be opened"


def test_walljump_4batch(get_od_model, get_cls_model, get_video_1, get_video_2, get_AddStreamModel):
    cap_1 = cv2.VideoCapture(get_video_1)
    cap_2 = cv2.VideoCapture(get_video_2)
    q = Queue(1)
    assert cap_1 is not None, "Failed to open video 1"
    assert cap_2 is not None, "Failed to open video 2"
    fps_1 = cap_1.get(cv2.CAP_PROP_FPS)
    fps_2 = cap_2.get(cv2.CAP_PROP_FPS)
    fps = max(fps_1, fps_2)

    user_params = []
    stream_ids = []
    for i in range(4):
        user_param_i, stream_id_i = get_AddStreamModel(i)
        user_params.append(user_param_i)
        stream_ids.append(stream_id_i)

    count = 0
    if cap_1.isOpened() and cap_2.isOpened():
        WalljumpService(q)
        while True:
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            if not ret_1 or not ret_2:
                break

            if count % round(fps * OD_TIME_INTERVAL_SECOND) == 0:
                q.put(
                    {
                        "batches": [frame_1, frame_1, frame_2, frame_2],
                        "stream_ids": stream_ids,
                        "user_params": user_params,
                    }
                )
                # cv2.imwrite("0.jpg", frame_1)
                # cv2.imwrite("1.jpg", frame_2)
                # time.sleep(OD_TIME_INTERVAL_SECOND - 0.01)
            count += 1

        stop_thread(q)
    else:
        assert False, "Video capture could not be opened"


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import WalljumpService

    assert WalljumpService is not None, "WalljumpService module import failed"
