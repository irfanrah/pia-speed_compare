import os
import cv2
import torch
import pytest
from queue import Queue
from pia_prod.AI.modules.khonkaen_peoplecounting.service import PeoplecountingService
from pia_prod.AI.modules.khonkaen_peoplecounting.config import (
    CROWD_PEOPLE_ONNX_MODEL_PATH,
    DEFAULT_PEOPLECOUNTING_INTERVAL,
)
from pia_prod.AI.modules.khonkaen_peoplecounting.param import PeoplecountingModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread


@pytest.fixture(scope="function")
def setup_pc_test(hf_downloader, nas_downloader, video_save_dir):
    """모델 다운로드 및 테스트 비디오 준비"""
    # 1. 모델 다운로드
    assert CROWD_PEOPLE_ONNX_MODEL_PATH is not None
    model_path = os.getenv("MODEL_CROWD_PEOPLE_ONNX_PATH", CROWD_PEOPLE_ONNX_MODEL_PATH)
    file_name = os.path.basename(model_path)
    repo_id = os.path.splitext(file_name)[0]
    hf_downloader.download(repo_id=repo_id, file_name=file_name, snapshot=False)

    # 2. 비디오 다운로드
    video_name = "DaeGu_crowd_people.mp4"
    local_path = os.path.join(video_save_dir, video_name)
    nas_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_path, local_path)

    return local_path


def create_pc_user_param(camera_id: str, threshold: int = 30):
    """테스트용 유저 파라미터 생성 헬퍼 함수"""
    return {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId=int(camera_id),
                cameraUrl="dummy_url",
                organization="pia",
                cvEvent=[
                    PeoplecountingModel(
                        name="peoplecounting_cv",
                        people_threshold=threshold,
                        roi=ROIModel(
                            roiId=1, polygonCoordinates=[950, 1509, 1612, 219, 21, 169, 17, 1511]
                        ),
                    ),
                ],
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }


# --- [Test 1] 기존: 단일 카메라 / Numpy 기반 배치 ---
def test_peoplecounting_batch_numpy(setup_pc_test):
    video_path = setup_pc_test
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)

    user_param = create_pc_user_param(camera_id="6")
    stream_id = "6_pia"

    if cap.isOpened():
        PeoplecountingService(q)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cnt += 1
            if cnt % round(fps * DEFAULT_PEOPLECOUNTING_INTERVAL) == 0:
                # Numpy 프레임 전달 (BGR)
                q.put({"batches": [frame], "stream_ids": [stream_id], "user_params": [user_param]})

        stop_thread(q)
    else:
        pytest.fail("Can't Open the Video")


# --- [Test 2] Torch Tensor 기반 단일 카메라 테스트 ---
def test_peoplecounting_torch_one_camera(setup_pc_test):
    video_path = setup_pc_test
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)

    user_param = create_pc_user_param(camera_id="10")
    stream_id = "10_pia"

    if cap.isOpened():
        PeoplecountingService(q)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cnt += 1
            if cnt % round(fps * DEFAULT_PEOPLECOUNTING_INTERVAL) == 0:
                # Numpy(BGR) -> Torch Tensor(RGB) 변환
                torch_frame = torch.from_numpy(frame)  # HWC, BGR, uint8
                torch_frame = torch_frame[:, :, [2, 1, 0]]  # HWC, RGB, uint8

                q.put(
                    {
                        "batches": [torch_frame],
                        "stream_ids": [stream_id],
                        "user_params": [user_param],
                    }
                )

        stop_thread(q)
    else:
        pytest.fail("Can't Open the Video")


# --- [Test 3] 멀티 카메라 (Numpy)
def test_peoplecounting_multi_np_batches(setup_pc_test):
    video_path = setup_pc_test
    cap = cv2.VideoCapture(video_path)  # 동일 영상을 두 카메라 인풋으로 가정
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)

    up1 = create_pc_user_param(camera_id="1")
    up2 = create_pc_user_param(camera_id="2")
    sid1, sid2 = "1", "2"

    if cap.isOpened():
        PeoplecountingService(q)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cnt += 1
            if cnt % round(fps * DEFAULT_PEOPLECOUNTING_INTERVAL) == 0:
                # 1번 인풋: Numpy
                frame_np = frame.copy()

                q.put(
                    {
                        "batches": [frame_np, frame_np],
                        "stream_ids": [sid1, sid2],
                        "user_params": [up1, up2],
                    }
                )

        stop_thread(q)
    else:
        pytest.fail("Can't Open the Video")


# --- [Test 4] 멀티 카메라 (Torch)
def test_peoplecounting_multi_torch_batches(setup_pc_test):
    video_path = setup_pc_test
    cap = cv2.VideoCapture(video_path)  # 동일 영상을 두 카메라 인풋으로 가정
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)

    up1 = create_pc_user_param(camera_id="1")
    up2 = create_pc_user_param(camera_id="2")
    sid1, sid2 = "1", "2"

    if cap.isOpened():
        PeoplecountingService(q)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cnt += 1
            if cnt % round(fps * DEFAULT_PEOPLECOUNTING_INTERVAL) == 0:
                # 1번 인풋: Torch Tensor
                torch_frame = torch.from_numpy(frame)  # HWC, BGR, uint8
                frame_torch = torch_frame[:, :, [2, 1, 0]]  # HWC, RGB, uint8
                q.put(
                    {
                        "batches": [frame_torch, frame_torch],
                        "stream_ids": [sid1, sid2],
                        "user_params": [up1, up2],
                    }
                )

        stop_thread(q)
    else:
        pytest.fail("Can't Open the Video")


# --- [Test 5] 멀티 카메라 (Torch) - 동적 배치 및 고유 ID 테스트 ---
@pytest.mark.parametrize("batch_size", [16])  # 여기서 테스트하고 싶은 배치 사이즈 조정
def test_peoplecounting_multi_torch_batches_param(setup_pc_test, batch_size):
    video_path = setup_pc_test
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    q = Queue(1)

    # 1. 고유한 카메라 ID 및 파라미터 생성 (Unique Int 기반)
    # camera_id: 1001, 1002, ..., 1016 와 같이 생성
    camera_ids = [1000 + i for i in range(1, batch_size + 1)]
    stream_ids = [f"{cid}_pia" for cid in camera_ids]
    user_params = [create_pc_user_param(camera_id=str(cid)) for cid in camera_ids]
    now_interval = 15  # 15초마다

    if cap.isOpened():
        # 서비스 시작
        PeoplecountingService(q)
        cnt = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cnt += 1
            # 설정된 간격(Interval)마다 큐에 데이터 삽입
            if cnt % round(fps * now_interval) == 0:
                # Numpy(BGR) -> Torch Tensor(RGB) 변환
                # (H, W, C) -> (C, H, W) 후 BGR -> RGB 채널 스왑
                torch_frame = torch.from_numpy(frame)  # HWC, BGR, uint8
                frame_torch = torch_frame[:, :, [2, 1, 0]]  # HWC, RGB, uint8

                # 동일 프레임을 batch_size만큼 복사하여 리스트 생성 (각각 다른 카메라 인풋 시뮬레이션)
                # 메모리 효율을 위해 .clone()은 필요시에만 사용 (Service 내부에서 슬라이싱하므로 원본 참조 가능)
                batch_frames = [frame_torch for _ in range(batch_size)]

                q.put(
                    {"batches": batch_frames, "stream_ids": stream_ids, "user_params": user_params}
                )

                # 대량 배치의 경우 추론 시간이 길어질 수 있으므로 첫 번째 배치 성공 후 중단하거나
                # 원하는 만큼 테스트를 지속하도록 설정 가능합니다.
                # break # 테스트 속도를 위해 한 배치만 확인하려면 주석 해제

        stop_thread(q)
    else:
        pytest.fail("Can't Open the Video")


def test_import_module():
    """Service 모듈 임포트 테스트"""
    from pia_prod.AI import PeoplecountingService

    assert PeoplecountingService is not None
