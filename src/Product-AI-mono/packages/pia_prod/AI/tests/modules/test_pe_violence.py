import pytest
import cv2
import os
from queue import Queue, Empty
import numpy as np

from pia_prod.AI.modules.pe_violence.service import PVService
from pia_prod.AI.modules.pe_violence.utils import draw_status
from pia_prod.AI.modules.pe_violence.config import (
    VIOLENCE_PE_ID,
    VIOLENCE_PE_MODEL_ONNX_PATH,
    VIOLENCE_PE_MODEL_TRT_PATH,
    VIOLENCE_PE_MODEL_PYTORCH_PATH,
    LIST_OF_NORMAL_TXT_PROMPTS,
    LIST_OF_VIOLENCE_TXT_PROMPTS,
    ALARM_QUEUE_SIZE,
    ALARM_THRESHOLD,
)
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import RetrievalBase
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.modules.pe_violence.trt_export import export_trt_engine
from pia_prod.AI.utils.init import logger
from pia_prod.AI.global_config import ALARMS_KEY

@pytest.fixture
def setup_model_and_video(hf_downloader, nas_downloader, video_save_dir):
    pt_name = os.path.basename(VIOLENCE_PE_MODEL_PYTORCH_PATH)
    onnx_name = os.path.basename(VIOLENCE_PE_MODEL_ONNX_PATH)
    hf_downloader.download(repo_id=VIOLENCE_PE_ID, file_name=pt_name, snapshot=False)
    hf_downloader.download(repo_id=VIOLENCE_PE_ID, file_name=onnx_name, snapshot=False)

    # Download normal and violence txt prompts
    for fname in LIST_OF_NORMAL_TXT_PROMPTS:
        hf_downloader.download(repo_id=VIOLENCE_PE_ID, file_name=fname, snapshot=False)

    for fname in LIST_OF_VIOLENCE_TXT_PROMPTS:
        hf_downloader.download(repo_id=VIOLENCE_PE_ID, file_name=fname, snapshot=False)

    export_trt_engine(
        onnx_file=VIOLENCE_PE_MODEL_ONNX_PATH,
        save_file_path=VIOLENCE_PE_MODEL_TRT_PATH,
        min_batch_size=1,
        opt_batch_size=4,
        max_batch_size=8,
    )

    video_name_1 = "pe_violence_1.mp4"
    video_name_2 = "pe_violence_2.mp4"

    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)

    debug_outvideo_path_1 = os.path.join(video_save_dir, "debug_outvideo_1.mp4")
    debug_outvideo_path_2 = os.path.join(video_save_dir, "debug_outvideo_2.mp4")

    nas_downloader.download_file(
        nas_downloader.get_nas_path(video_name_1),
        local_video_path_1,
    )
    nas_downloader.download_file(
        nas_downloader.get_nas_path(video_name_2),
        local_video_path_2,
    )

    return {
        "video1": local_video_path_1,
        "video2": local_video_path_2,
        "debug1": debug_outvideo_path_1,
        "debug2": debug_outvideo_path_2,
    }


def _make_user_param(camera_id: str, camera_url: str, organization: str):
    """
    pe_violence는 정상/폭력 2클래스로 동작하고 compare_embedding 방식으로
    config 내 NORMAL/VIOLENCE 벡터 2개만 사용합니다.
    카메라마다 서로 다른 ROI 상태를 구성하여 ROI 처리 로직을 검증한다.
    """
    if camera_id == "7":
        polygon_coordinates = []  # ROI 없음, 감지되어야 함
    else:
        polygon_coordinates = [
            120,
            90,
            1820,
            95,
            1790,
            1070,
            110,
            1050,
        ]  # 샘플 ROI 좌표 (사각형 형태)

    return {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId=camera_id,
                cameraUrl=camera_url,
                organization=organization,
                retEvent=[
                    RetrievalBase(
                        name="violence_ret",
                        incidentThresholdSecond=3,
                        incidentTimeoutSecond=3,
                        roi=ROIModel(
                            roiId=1,
                            polygonCoordinates=polygon_coordinates,
                        ),
                        topCandidates=1,
                        abnormalText=["violence"],
                        normalText=["normal"],
                    ),
                ],
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }


def test_deterministic_pe_violence_single_camera(setup_model_and_video):
    """
    Deterministic test (single-thread):
    - Enqueues inputs via Queue.put() (matches real pipeline shape)
    - Consumes exactly one item at a time from the queue (no background threads)
    - Produces a debug video and asserts at least one decision is produced
    """
    paths = setup_model_and_video
    video_path = paths["video1"]
    debug_out_path = paths["debug1"]

    status_text = "NORMAL"
    alarm_queue_count = 0
    alarm_queue_entries = []
    is_triggered = False

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps and fps > 0, f"Invalid FPS from video: {fps}"

    # Queue is the only input path (like real processing)
    q = Queue(maxsize=1)

    service = PVService(q)
    service.debug = True
    service._load_model()
    service._load_roi_manager()
    service._load_event_manager()

    # Match PVService.num_gather_frames (fps 3) for frame sampling interval
    interval = max(1, round(fps / service.num_gather_frames))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_out_path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter: {debug_out_path}"

    user_param = _make_user_param(camera_id="7", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    decisions = []  # store (frame_count, alarms_dict)

    def _process_one_from_queue():
        """
        Single-step queue consumer that mimics the production pipeline:
        - get() payload from queue
        - call service detection using the payload content
        Returns detect_out (None or tuple) exactly like _detect().
        """
        try:
            item = q.get_nowait()
        except Empty:
            return None

        # IMPORTANT: queue.task_done() only if you use join(); safe to call anyway
        q.task_done()

        return service._detect(
            batches=item["batches"],
            stream_ids=item["stream_ids"],
            user_params=item["user_params"],
            fps=item["fps"],
        )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis_frame = frame.copy()

        # deterministic sanity checks
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8

        if frame_count % interval == 0:
            payload = {
                "batches": [frame],
                "stream_ids": [stream_id],
                "user_params": [user_param],
                "fps": fps,
            }

            # ONLY input method: Queue.put()
            # If queue is full (shouldn't be in this single-thread pattern), drop oldest then put.
            if q.full():
                try:
                    _ = q.get_nowait()
                    q.task_done()
                except Empty:
                    pass
            q.put(payload)

            # Consume exactly one item (still deterministic, no threads)
            detect_out = _process_one_from_queue()

            # Read alarm queue state directly from event manager
            eq = service.alarm_event_manager.duration_queue.get(stream_id)
            if eq is not None:
                alarm_queue_entries = list(eq)
                alarm_queue_count = int(sum(eq))
                is_triggered = alarm_queue_count >= service.alarm_event_manager.alarm_threshold

            if detect_out is not None:
                alarms = detect_out[ALARMS_KEY]
                decisions.append((frame_count, alarms))

                val = alarms.get(stream_id, [None, None])[0]
                if val is True:
                    status_text = "VIOLENCE"
                elif val is False:
                    status_text = "NORMAL"

                logger.info(
                    f"[frame={frame_count}] status={status_text} "
                    f"queue={alarm_queue_count}/{ALARM_QUEUE_SIZE} "
                    f"thr={ALARM_THRESHOLD} triggered={is_triggered}"
                )

        draw_status(
            vis_frame,
            status_text,
            frame_count,
            width,
            height,
            alarm_queue_count=alarm_queue_count,
            alarm_queue_size=ALARM_QUEUE_SIZE,
            alarm_threshold=ALARM_THRESHOLD,
            is_triggered=is_triggered,
            alarm_queue_entries=alarm_queue_entries,
        )
        writer.write(vis_frame)

    cap.release()
    writer.release()

    logger.info(f"Detected {len(decisions)} decisions.")
    assert (
        len(decisions) > 0
    ), "PVService never produced a decision (True/False). (No Anomaly found)"


def test_pe_violence_single_camera(setup_model_and_video):
    """
    단일 스트림에 대해 프레임을 일정 간격으로 큐에 넣고, PVService가 스레드에서 추론/이벤트 로직을 수행하는지 확인한다.
    단일 스트림에 대해 ROI가 비어 있는 상태, ROI가 지정된 상태에서도 PVService가 정상 동작하는지 확인한다.한
    """
    paths = setup_model_and_video
    video_path = paths["video1"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = round(fps / 8)
    q = Queue(1)

    user_param = _make_user_param(camera_id="6", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    if cap.isOpened():
        PVService(q)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % interval == 0:
                q.put(
                    {
                        "batches": [frame],
                        "stream_ids": [stream_id],
                        "user_params": [user_param],
                        "fps": fps,
                    }
                )

        # 서비스 종료 및 스레드 정리
        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_pe_violence_multiple_cameras(setup_model_and_video):
    """
    두 개의 카메라 스트림에 대해 서로 다른 ROI 상태를 적용하여 멀티 스트림 경로를 검증한다.
    """
    paths = setup_model_and_video
    video_path_1 = paths["video1"]
    video_path_2 = paths["video2"]
    q = Queue(1)
    camera_specs = [
        {
            "video_path": video_path_1,
            "user_param": _make_user_param(camera_id="6", camera_url="11", organization="pia"),
        },
        {
            "video_path": video_path_2,
            "user_param": _make_user_param(camera_id="7", camera_url="12", organization="pia"),
        },
        {
            "video_path": video_path_1,
            "user_param": _make_user_param(camera_id="8", camera_url="13", organization="pia"),
        },
        {
            "video_path": video_path_2,
            "user_param": _make_user_param(camera_id="9", camera_url="14", organization="pia"),
        },
    ]

    probe_cap = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe_cap.isOpened(), "Can't Open the Video"
        fps = int(probe_cap.get(cv2.CAP_PROP_FPS))
    finally:
        probe_cap.release()

    interval = max(1, round(fps / 8))
    PVService(q)

    stream_contexts = []
    for spec in camera_specs:
        cap = cv2.VideoCapture(spec["video_path"])
        assert cap.isOpened(), "Can't Open the Video"
        payload = spec["user_param"]["user_param"]
        stream_id = f"{payload['cameraId']}_{payload['organization']}"
        stream_contexts.append(
            {
                "cap": cap,
                "stream_id": stream_id,
                "user_param": spec["user_param"],
                "frame_count": 0,
            }
        )

    stream_order = [ctx["stream_id"] for ctx in stream_contexts]
    user_param_map = {ctx["stream_id"]: ctx["user_param"] for ctx in stream_contexts}
    pending_frames = {stream_id: None for stream_id in stream_order}

    try:
        while True:
            frames_ready = True
            for ctx in stream_contexts:
                ret, frame = ctx["cap"].read()
                if not ret:
                    frames_ready = False
                    break

                ctx["frame_count"] += 1
                if ctx["frame_count"] % interval == 0:
                    pending_frames[ctx["stream_id"]] = frame

            if not frames_ready:
                break

            if all(pending_frames[stream_id] is not None for stream_id in stream_order):
                frames = [pending_frames[stream_id] for stream_id in stream_order]
                user_params = [user_param_map[stream_id] for stream_id in stream_order]
                q.put(
                    {
                        "batches": frames,
                        "stream_ids": stream_order,
                        "user_params": user_params,
                        "fps": fps,
                    }
                )
                for stream_id in stream_order:
                    pending_frames[stream_id] = None
    finally:
        for ctx in stream_contexts:
            ctx["cap"].release()

    stop_thread(q)


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import PVService

    assert PVService is not None, "PVService module import failed"


def test_pe_violence_torch_one_camera(setup_model_and_video):
    """
    단일 카메라 스트림 테스트:
    Input Data를 numpy array가 아닌 torch.Tensor (H, W, C, uint8) 형태로 변환하여
    PVService가 Tensor 입력을 정상적으로 처리하는지 검증한다.
    """
    import torch

    paths = setup_model_and_video
    video_path = paths["video1"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = round(fps / 8)
    q = Queue(1)

    user_param = _make_user_param(camera_id="11", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    if cap.isOpened():
        # 서비스 시작
        PVService(q)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % interval == 0:
                # [Requirement 1] numpy(cv2) -> torch tensor [H,W,C]
                # 정규화 하지 않음 (0-255 uint8 유지), 차원 변경 없음
                torched_frame = torch.from_numpy(frame)
                torched_frame = torched_frame[..., [2, 1, 0]]  # BGR -> RGB

                q.put(
                    {
                        "batches": [torched_frame],  # List of Tensors
                        "stream_ids": [stream_id],
                        "user_params": [user_param],
                        "fps": fps,
                    }
                )

        # 서비스 종료 및 스레드 정리
        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_pe_violence_torch_two_cameras(setup_model_and_video):
    """
    멀티 카메라(2개) 스트림 테스트:
    두 개의 스트림에서 나온 프레임을 모두 torch.Tensor (H, W, C)로 변환한 뒤,
    배치 리스트에 담아 PVService로 전달하여 멀티 배치 텐서 처리를 검증한다.
    """
    import torch

    paths = setup_model_and_video
    video_path_1 = paths["video1"]
    video_path_2 = paths["video2"]
    q = Queue(1)

    camera_specs = [
        {
            "video_path": video_path_1,
            "user_param": _make_user_param(camera_id="12", camera_url="22", organization="pia"),
        },
        {
            "video_path": video_path_2,
            "user_param": _make_user_param(camera_id="13", camera_url="33", organization="pia"),
        },
    ]

    # FPS 확인용 프로브
    probe_cap = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe_cap.isOpened(), "Can't Open the Video"
        fps = int(probe_cap.get(cv2.CAP_PROP_FPS))
    finally:
        probe_cap.release()

    interval = max(1, round(fps / 8))
    PVService(q)

    # 스트림 컨텍스트 초기화
    stream_contexts = []
    for spec in camera_specs:
        cap = cv2.VideoCapture(spec["video_path"])
        assert cap.isOpened(), f"Can't Open the Video: {spec['video_path']}"

        payload = spec["user_param"]["user_param"]
        stream_id = f"{payload['cameraId']}_{payload['organization']}"

        stream_contexts.append(
            {
                "cap": cap,
                "stream_id": stream_id,
                "user_param": spec["user_param"],
                "frame_count": 0,
            }
        )

    stream_order = [ctx["stream_id"] for ctx in stream_contexts]
    user_param_map = {ctx["stream_id"]: ctx["user_param"] for ctx in stream_contexts}
    pending_frames = {stream_id: None for stream_id in stream_order}

    try:
        while True:
            frames_ready = True
            # 각 카메라에서 프레임 읽기
            for ctx in stream_contexts:
                ret, frame = ctx["cap"].read()
                if not ret:
                    frames_ready = False
                    break

                ctx["frame_count"] += 1
                if ctx["frame_count"] % interval == 0:
                    frame = frame[..., [2, 1, 0]]  # BGR -> RGB
                    pending_frames[ctx["stream_id"]] = frame

            if not frames_ready:
                break

            # 두 카메라의 프레임이 모두 준비되었을 때 배치 처리
            if all(pending_frames[stream_id] is not None for stream_id in stream_order):
                # 원본 프레임 리스트
                raw_frames = [pending_frames[stream_id] for stream_id in stream_order]

                # [Requirement 1] Convert all frames in the batch to Torch Tensors [H, W, C]
                torched_batch = [torch.from_numpy(f) for f in raw_frames]

                user_params = [user_param_map[stream_id] for stream_id in stream_order]

                q.put(
                    {
                        "batches": torched_batch,  # List of Tensors
                        "stream_ids": stream_order,
                        "user_params": user_params,
                        "fps": fps,
                    }
                )

                # 대기열 초기화
                for stream_id in stream_order:
                    pending_frames[stream_id] = None

    finally:
        for ctx in stream_contexts:
            ctx["cap"].release()

    stop_thread(q)
