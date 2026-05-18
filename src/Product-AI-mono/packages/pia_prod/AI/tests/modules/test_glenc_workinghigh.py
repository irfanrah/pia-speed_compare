import os
import time
from queue import Queue

import cv2
import pytest

from pia.ai.exports.trt import onnx2trt
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time

from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.modules.glenc_workinghigh.config import (
    PERSON_DETECTION_MODEL_ONNX_PATH,
    OD_INPUT_SIZE,
    OD_TIME_INTERVAL_SECOND,
    IMAGE_SAVE_PATH,
)
from pia_prod.AI.modules.glenc_workinghigh.param import WorkinghighModel
from pia_prod.AI.modules.glenc_workinghigh.service import WorkinghighService
from pia_prod.AI.modules.khonkaen_weapon.debug_utils import VideoSaveManager
from pia_prod.AI.tests.conftest import set_logging_flag
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.utils.utils import AddStreamModel2dict


@pytest.fixture(scope="function")
def model_download(hf_downloader):
    assert PERSON_DETECTION_MODEL_ONNX_PATH is not None, "Please, check model file path"
    model_path = PERSON_DETECTION_MODEL_ONNX_PATH
    file_name = os.path.basename(model_path)
    repo_id, _ = os.path.splitext(file_name)
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)
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


@pytest.fixture(scope="function")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "workinghigh_1.mp4"
    local_video_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_video_path)
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
                WorkinghighModel(
                    name="workinghigh_cv",
                    incidentThresholdSecond=3,
                    incidentTimeoutSecond=3,
                    confidenceThreshold=0.5,
                    nmsThreshold=0.5,
                    roi=ROIModel(
                        roiId=1,
                        polygonCoordinates=[562,491,765,669,940,601,793,460],
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


def test_glenc_workinghigh(model_download, local_video_path, get_AddStreamModel):
    cap = cv2.VideoCapture(local_video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = 0.3
        q = Queue(1)

        user_param, stream_id = get_AddStreamModel(1)

        video_speed = 2.0  # 비디오 재생 속도
        model_input_second = OD_TIME_INTERVAL_SECOND / video_speed  # 원래는 OD_TIME_INTERVAL_SECOND
        count = 0

        if cap.isOpened():
            service = WorkinghighService(q)
            service.logging_flag = set_logging_flag()
            service.save_video = True
            service.video_manager = VideoSaveManager(
                output_dir=os.path.join(IMAGE_SAVE_PATH, stream_id, "workinghigh"),
                save_video=service.save_video,
                model_input_second=model_input_second,
                save_interval_seconds=10,
                prefix="video",
            )

            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % round(fps * interval) == 0:
                    q.put(
                        {"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]}
                    )
                count += 1

            service.video_manager.close()
            service.video_manager = None
            stop_thread(q)
        else:
            assert False, "Can't Open the Video"
    finally:
        cap.release()


def test_glenc_workinghigh_batches(
    model_download, local_video_path, get_AddStreamModel, batch_size=4
):
    cap_1 = cv2.VideoCapture(local_video_path)
    try:
        q = Queue(1)
        assert cap_1 is not None, "Failed to open video 1"

        user_params = []
        stream_ids = []
        for i in range(batch_size):
            user_param_i, stream_id_i = get_AddStreamModel(i)
            user_params.append(user_param_i)
            stream_ids.append(stream_id_i)

        count = 0
        fps = cap_1.get(cv2.CAP_PROP_FPS)
        if cap_1.isOpened():
            service = WorkinghighService(q)
            service.logging_flag = set_logging_flag()

            while True:
                ret_1, frame_1 = cap_1.read()
                if not ret_1:
                    break

                batches = [frame_1] * batch_size
                assert len(batches) == batch_size, f"Batch size should be {batch_size}"

                if count % round(fps * OD_TIME_INTERVAL_SECOND) == 0:
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

            stop_thread(q)
        else:
            assert False, "Video capture could not be opened"
    finally:
        cap_1.release()


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import WorkinghighService

    assert WorkinghighService is not None, "WorkinghighService module import failed"
