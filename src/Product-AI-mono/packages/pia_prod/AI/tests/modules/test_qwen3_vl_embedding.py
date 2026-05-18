import pytest
import cv2
import os
import torch
from queue import Queue, Empty
import numpy as np

from pia_prod.AI.modules.ft_pe.utils import draw_multi_category_status
from pia_prod.AI.modules.qwen3_vl_embedding.service import Qwen3VLEService
from pia_prod.AI.modules.qwen3_vl_embedding.config import (
    CATEGORY_EVENT_MAP,
    QWEN3VLE_ID,
)
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import RetrievalBase
from pia_prod.AI.DTO.param_base import ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.utils.init import logger
from pia_prod.AI.global_config import ALARMS_KEY

@pytest.fixture(scope="session")
def setup_model_and_video(hf_downloader, nas_downloader, video_save_dir):
    hf_downloader.download(
        repo_id=QWEN3VLE_ID,
        snapshot=True,
    )

    video_name_1 = "qwen3vle_fire.mp4"
    video_name_2 = "pe_violence_1.mp4"

    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)

    debug_outvideo_path_1 = os.path.join(video_save_dir, "debug_qwen3vle_out_1.mp4")
    debug_outvideo_path_2 = os.path.join(video_save_dir, "debug_qwen3vle_out_2.mp4")

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
    Qwen3 VL Embedding supports multiple categories: fire, falldown, violence, smoke.
    Each category has independent ROI and event configuration.
    Configure different ROI states for each camera to verify ROI processing logic.
    """
    if camera_id == "7":
        polygon_coordinates = []  # No ROI, should be detected
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
        ]  # Sample ROI coordinates (rectangular shape)

    # Create retrieval events for all categories. The authoritative list lives
    # in config.CATEGORY_EVENT_MAP — adding an entry there auto-registers it here.
    ret_events = []
    for category in CATEGORY_EVENT_MAP:
        ret_events.append(
            RetrievalBase(
                name=f"{category}_vle_ret",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                roi=ROIModel(
                    roiId=1,
                    polygonCoordinates=polygon_coordinates,
                ),
                topCandidates=1,
                abnormalText=[category],
                normalText=["normal"],
            )
        )

    return {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId=camera_id,
                cameraUrl=camera_url,
                organization=organization,
                retEvent=ret_events,
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }


def test_deterministic_qwen3_vl_single_camera(setup_model_and_video):
    """
    Deterministic test (single-thread) for Qwen3 VL Embedding:
    - Enqueues inputs via Queue.put() (matches real pipeline shape)
    - Consumes exactly one item at a time from the queue (no background threads)
    - Produces a debug video and asserts at least one decision is produced
    - Validates multi-category predictions (fire, falldown, violence, smoke)
    """
    paths = setup_model_and_video
    video_path = paths["video1"]
    debug_out_path = paths["debug1"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps and fps > 0, f"Invalid FPS from video: {fps}"
    interval = max(1, round(fps / 2))

    # Queue is the only input path (like real processing)
    q = Queue(maxsize=1)
    service = Qwen3VLEService(q)
    service.debug = True
    em = service.alarm_event_manager  # shortcut — used for per-frame overlay

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_out_path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter: {debug_out_path}"

    user_param = _make_user_param(camera_id="7", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    # Authoritative category list lives in config.CATEGORY_EVENT_MAP.
    # The `_vle_ret` suffix here matches the names _make_user_param registers.
    categories = list(CATEGORY_EVENT_MAP.keys())
    ret_event_ids = [f"{cat}_vle_ret" for cat in categories]

    decisions = []  # store (frame_count, alarms_dict)
    status_rows = []  # (label, queue_list, triggered) — refreshed on detection cycles

    def _process_one_from_queue():
        """
        Single-step queue consumer that mimics the production pipeline:
        - get() payload from queue
        - call service detection using the payload content
        Returns detect_out (None or dict) exactly like _detect().
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

            # ✅ ONLY input method: Queue.put()
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

            if detect_out is not None:
                alarms = detect_out[ALARMS_KEY]
                decisions.append((frame_count, alarms))

                # Multi-category: alarms are keyed by f"{stream_id}__{category_id}",
                # so one stream can fire several keys in the same frame.
                prefix = f"{stream_id}__"
                detected_categories = [
                    key[len(prefix):] for key in alarms if key.startswith(prefix)
                ]
                if detected_categories:
                    logger.info(f"Frame {frame_count} detected: {detected_categories}")

            # Snapshot per-category queue state for the overlay renderer.
            status_rows = []
            for label, ret_id in zip(categories, ret_event_ids):
                queue = list(em.duration_queue[stream_id][ret_id])
                triggered = sum(queue) >= em.alarm_duration
                status_rows.append((label, queue, triggered))

        draw_multi_category_status(
            vis_frame,
            frame_count,
            width,
            height,
            status_rows,
            queue_size=em.queue_size,
            threshold=em.alarm_duration,
        )
        writer.write(vis_frame)

    cap.release()
    writer.release()

    logger.info(f"Detected {len(decisions)} decisions.")
    assert (
        len(decisions) > 0
    ), "Qwen3VLEService never produced a decision. (No Anomaly found)"

    # Service shutdown and GPU cleanup
    stop_thread(q)


def test_qwen3_vl_single_camera(setup_model_and_video):
    """
    For a single stream, put frames at regular intervals into the queue and verify that Qwen3VLEService performs inference/event logic in a thread.
    Verify that multi-category (fire, falldown, violence, smoke) predictions are processed correctly.
    """
    paths = setup_model_and_video
    video_path = paths["video1"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 2))
    q = Queue(1)

    user_param = _make_user_param(camera_id="6", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    if cap.isOpened():
        Qwen3VLEService(q)
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

        # Service shutdown and thread cleanup
        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_qwen3_vl_multiple_cameras(setup_model_and_video):
    """
    For two camera streams, apply different ROI states to verify the multi-stream path.
    Verify that multi-category predictions are processed independently per stream.
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

    interval = max(1, round(fps / 2))
    Qwen3VLEService(q)

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
            # Read frames from each camera
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

            # Batch processing when all frames are ready
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
    from pia_prod.AI import Qwen3VLEService

    assert Qwen3VLEService is not None, "Qwen3VLEService module import failed"


def test_qwen3_vl_torch_one_camera(setup_model_and_video):
    """
    Single camera stream test:
    Convert Input Data to torch.Tensor (H, W, C, uint8) format rather than numpy array
    and verify that Qwen3VLEService handles Tensor input correctly.
    Verify that multi-category predictions are also handled correctly with torch tensor input.
    """
    paths = setup_model_and_video
    video_path = paths["video1"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 2))
    q = Queue(1)

    user_param = _make_user_param(camera_id="11", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    if cap.isOpened():
        # Service start
        Qwen3VLEService(q)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % interval == 0:
                # [Requirement 1] numpy(cv2) -> torch tensor [H,W,C]
                # No normalization (maintain 0-255 uint8), no dimension changes
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

        # Service shutdown and thread cleanup
        stop_thread(q)
    else:
        assert False, "Can't Open the Video"


def test_qwen3_vl_torch_two_cameras(setup_model_and_video):
    """
    Multi-camera (2) stream test:
    Convert frames from two streams to torch.Tensor (H, W, C), put them in a batch list,
    and pass to Qwen3VLEService to verify multi-batch tensor processing.
    Verify that multi-category multi-stream torch input is handled correctly.
    """
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

    interval = max(1, round(fps / 2))
    Qwen3VLEService(q)

    # Stream context initialization
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
            # Read frames from each camera
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

            # Batch processing when frames from both cameras are ready
            if all(pending_frames[stream_id] is not None for stream_id in stream_order):
                # Original frame list
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

                # Queue initialization
                for stream_id in stream_order:
                    pending_frames[stream_id] = None

    finally:
        for ctx in stream_contexts:
            ctx["cap"].release()

    stop_thread(q)