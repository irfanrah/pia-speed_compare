import pytest
import cv2
import os
import torch
from queue import Queue, Empty

from pia_prod.AI.modules.pe_vle_2stage_sync.service import PeVle2StageSyncService
from pia_prod.AI.modules.qwen3_vl_embedding.utils import draw_status
from pia_prod.AI.modules.perception_encoder.config import (
    PERCEPTION_ENCODER_TXT_FEATURE_PATH,
    PERCEPTION_ENCODER_TRT_PATH,
    PERCEPTION_ENCODER_ONNX_PATH,
    PERCEPTION_ENCODER_PYTORCH_PATH,
    INPUT_SIZE,
)
from pia_prod.AI.modules.perception_encoder.trt_export import export_trt_engine
from pia_prod.AI.modules.qwen3vle_trt.config import (
    QWEN3VLE_TRT_ID,
    QWEN3VLE_TRT_ONNX_DIR_PATH,
)
from pia_prod.AI.modules.qwen3vle_trt.export.c_export_onnx_to_trt import export_to_trt
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.DTO.param_base import RetrievalBase, ROIModel
from pia_prod.AI.utils.utils import AddStreamModel2dict
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.utils.init import logger
from pia_prod.AI.global_config import ALARMS_KEY


@pytest.fixture(scope="session")
def setup_models_and_video(hf_downloader, nas_downloader, video_save_dir):
    """
    Downloads assets for Qwen3 (and PE if hosted similarly) and test videos.
    """
    # PE Assets
    txt_json_name = os.path.basename(PERCEPTION_ENCODER_TXT_FEATURE_PATH)
    onnx_file_save_path = PERCEPTION_ENCODER_ONNX_PATH
    repo_name = os.path.splitext(os.path.basename(PERCEPTION_ENCODER_PYTORCH_PATH))[0]
    onnx_repo_name = os.path.splitext(os.path.basename(onnx_file_save_path))[0]
    download_model_name = os.path.basename(onnx_file_save_path)

    hf_downloader.download(repo_id=onnx_repo_name, file_name=download_model_name, snapshot=False)
    hf_downloader.download(repo_id=repo_name, file_name=txt_json_name, snapshot=False)
    export_trt_engine(
        onnx_file=onnx_file_save_path,
        save_file_path=PERCEPTION_ENCODER_TRT_PATH,
        min_batch_size=1,
        opt_batch_size=4,
        max_batch_size=8,
        input_size=INPUT_SIZE,
    )

    # Qwen3VLE TRT Assets — download the ONNX/engine repo and (re)build engines
    # only if missing in QWEN3VLE_TRT_ONNX_DIR_PATH; otherwise this is a no-op.
    hf_downloader.download(repo_id=QWEN3VLE_TRT_ID, snapshot=True)
    export_to_trt(onnx_dir=QWEN3VLE_TRT_ONNX_DIR_PATH)

    # Video Assets
    video_name_1 = "qwen3vle_fire.mp4"
    video_name_2 = "two_stage_pe_qwen3vle_normal.mp4"

    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)

    debug_outvideo_path_1 = os.path.join(video_save_dir, "debug_twostage_out_1.mp4")
    debug_outvideo_path_2 = os.path.join(video_save_dir, "debug_twostage_out_2.mp4")

    nas_downloader.download_file(nas_downloader.get_nas_path(video_name_1), local_video_path_1)
    nas_downloader.download_file(nas_downloader.get_nas_path(video_name_2), local_video_path_2)

    return {
        "video1": local_video_path_1,
        "video2": local_video_path_2,
        "debug1": debug_outvideo_path_1,
        "debug2": debug_outvideo_path_2,
    }


def _make_user_param(camera_id: str, camera_url: str, organization: str):
    if camera_id == "7":
        polygon_coordinates = [] 
    else:
        polygon_coordinates = [120, 90, 1820, 95, 1790, 1070, 110, 1050]

    ret_events = []
    for category in ["fire", "smoke"]:
        ret_events.append(
            RetrievalBase(
                name=f"{category}_pe_vle_ret",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                roi=ROIModel(roiId=1, polygonCoordinates=polygon_coordinates),
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


def test_import_two_stage_pe_qwen3vle_module():
    """Test Importing the Two Stage Service Module."""
    from pia_prod.AI import PeVle2StageSyncService
    assert PeVle2StageSyncService is not None, "PE Qwen3VLE 2 Stage module import failed"


def test_deterministic_two_stage_pe_qwen3vle_single_camera_anomaly_case(setup_models_and_video):
    """
    Deterministic test to ensure the 2-stage pipeline successfully routes 
    flagged frames from PE to Qwen3 for TP case.
    """
    paths = setup_models_and_video
    video_path = paths["video1"]
    debug_out_path = paths["debug1"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 2))

    q = Queue(maxsize=1)
    service = PeVle2StageSyncService(q)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_out_path, fourcc, fps, (width, height))

    user_param = _make_user_param(camera_id="7", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    decisions = []
    status_text = "NORMAL"
    frame_count = 0

    def _process_one_from_queue():
        try:
            item = q.get_nowait()
        except Empty:
            return None
        q.task_done()
        return service._detect(
            batches=item["batches"],
            stream_ids=item["stream_ids"],
            user_params=item["user_params"],
            fps=item["fps"],
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis_frame = frame.copy()

        if frame_count % interval == 0:
            payload = {
                "batches": [frame],
                "stream_ids": [stream_id],
                "user_params": [user_param],
                "fps": fps,
            }

            if q.full():
                try:
                    _ = q.get_nowait()
                    q.task_done()
                except Empty:
                    pass
            q.put(payload)

            detect_out = _process_one_from_queue()

            if detect_out is not None and ALARMS_KEY in detect_out:
                alarms = detect_out[ALARMS_KEY]
                decisions.append((frame_count, alarms))

                if stream_id in alarms:
                    category_detected = alarms[stream_id][1] 
                    status_text = f"ANOMALY DETECTED: {category_detected}"
                else:
                    status_text = "NORMAL"

                logger.info(f"Update debug video status: {status_text}")

        draw_status(vis_frame, status_text, frame_count, width, height)
        writer.write(vis_frame)

    cap.release()
    writer.release()

    logger.info(f"Detected {len(decisions)} anomalies.")
    assert (
        len(decisions) > 0
    ), "No Anomaly found in the Fire Video!"

    # Service shutdown and GPU cleanup
    stop_thread(q)


def test_deterministic_two_stage_pe_qwen3vle_single_camera_normal_case(setup_models_and_video):
    """
    Deterministic test to ensure the 2-stage pipeline successfully routes 
    flagged frames from PE to Qwen3 for FP case in PE.
    """
    paths = setup_models_and_video
    video_path = paths["video2"]
    debug_out_path = paths["debug2"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 2))

    q = Queue(maxsize=1)
    service = PeVle2StageSyncService(q)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_out_path, fourcc, fps, (width, height))

    user_param = _make_user_param(camera_id="7", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    decisions = []
    status_text = "NORMAL"
    frame_count = 0

    def _process_one_from_queue():
        try:
            item = q.get_nowait()
        except Empty:
            return None
        q.task_done()
        return service._detect(
            batches=item["batches"],
            stream_ids=item["stream_ids"],
            user_params=item["user_params"],
            fps=item["fps"],
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis_frame = frame.copy()

        if frame_count % interval == 0:
            payload = {
                "batches": [frame],
                "stream_ids": [stream_id],
                "user_params": [user_param],
                "fps": fps,
            }

            if q.full():
                try:
                    _ = q.get_nowait()
                    q.task_done()
                except Empty:
                    pass
            q.put(payload)

            detect_out = _process_one_from_queue()

            if detect_out is not None and ALARMS_KEY in detect_out:
                alarms = detect_out[ALARMS_KEY]
                decisions.append((frame_count, alarms))

                if stream_id in alarms:
                    category_detected = alarms[stream_id][1] 
                    status_text = f"ANOMALY DETECTED: {category_detected}"
                else:
                    status_text = "NORMAL"

                logger.info(f"Update debug video status: {status_text}")

        draw_status(vis_frame, status_text, frame_count, width, height)
        writer.write(vis_frame)

    cap.release()
    writer.release()

    logger.info(f"Detected {len(decisions)} anomalies.")
    assert (
        len(decisions) == 0
    ), "Anomaly found in the Normal Video!"

    # Service shutdown and GPU cleanup
    stop_thread(q)


def test_two_stage_pe_qwen3vle_multiple_cameras(setup_models_and_video):
    """
    Test the multi-stream pipeline with different ROI states to ensure 
    the service successfully handles arrays of batches and partial filtering.
    """
    paths = setup_models_and_video
    video_path_1 = paths["video1"]
    video_path_2 = paths["video2"]
    q = Queue(1)
    
    camera_specs = [
        {"video_path": video_path_1, "user_param": _make_user_param(camera_id="6", camera_url="11", organization="pia")},
        {"video_path": video_path_2, "user_param": _make_user_param(camera_id="7", camera_url="12", organization="pia")},
    ]

    probe_cap = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe_cap.isOpened()
        fps = int(probe_cap.get(cv2.CAP_PROP_FPS))
    finally:
        probe_cap.release()

    interval = max(1, round(fps / 2))
    PeVle2StageSyncService(q)

    stream_contexts = []
    for spec in camera_specs:
        cap = cv2.VideoCapture(spec["video_path"])
        payload = spec["user_param"]["user_param"]
        stream_id = f"{payload['cameraId']}_{payload['organization']}"
        stream_contexts.append({
            "cap": cap,
            "stream_id": stream_id,
            "user_param": spec["user_param"],
            "frame_count": 0,
        })

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
                
                q.put({
                    "batches": frames,
                    "stream_ids": stream_order,
                    "user_params": user_params,
                    "fps": fps,
                })
                
                for stream_id in stream_order:
                    pending_frames[stream_id] = None
    finally:
        for ctx in stream_contexts:
            ctx["cap"].release()

    stop_thread(q)


def test_two_stage_pe_qwen3vle_torch_one_camera(setup_models_and_video):
    """
    Single camera stream test with torch.Tensor (H, W, C, uint8, RGB) input.
    Verifies that the two-stage pipeline handles decoded torch frames
    from an external inference server (e.g. ES team).
    """
    paths = setup_models_and_video
    video_path = paths["video1"]
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / 2))
    q = Queue(1)

    user_param = _make_user_param(camera_id="7", camera_url="11", organization="pia")
    stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"

    PeVle2StageSyncService(q)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % interval == 0:
            torched_frame = torch.from_numpy(frame)
            torched_frame = torched_frame[..., [2, 1, 0]]  # BGR -> RGB

            q.put({
                "batches": [torched_frame],
                "stream_ids": [stream_id],
                "user_params": [user_param],
                "fps": fps,
            })

    cap.release()
    stop_thread(q)


def test_two_stage_pe_qwen3vle_torch_two_cameras(setup_models_and_video):
    """
    Multi-camera (2) stream test with torch.Tensor (H, W, C, uint8, RGB) input.
    Verifies that the two-stage pipeline handles batched torch frames
    from multiple streams simultaneously.
    """
    paths = setup_models_and_video
    video_path_1 = paths["video1"]
    video_path_2 = paths["video2"]
    q = Queue(1)

    camera_specs = [
        {"video_path": video_path_1, "user_param": _make_user_param(camera_id="8", camera_url="11", organization="pia")},
        {"video_path": video_path_2, "user_param": _make_user_param(camera_id="9", camera_url="12", organization="pia")},
    ]

    probe_cap = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe_cap.isOpened()
        fps = int(probe_cap.get(cv2.CAP_PROP_FPS))
    finally:
        probe_cap.release()

    interval = max(1, round(fps / 2))
    PeVle2StageSyncService(q)

    stream_contexts = []
    for spec in camera_specs:
        cap = cv2.VideoCapture(spec["video_path"])
        assert cap.isOpened(), f"Can't open video: {spec['video_path']}"
        payload = spec["user_param"]["user_param"]
        stream_id = f"{payload['cameraId']}_{payload['organization']}"
        stream_contexts.append({
            "cap": cap,
            "stream_id": stream_id,
            "user_param": spec["user_param"],
            "frame_count": 0,
        })

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
                torched_batch = [
                    torch.from_numpy(pending_frames[sid][..., [2, 1, 0]])  # BGR -> RGB
                    for sid in stream_order
                ]
                user_params = [user_param_map[stream_id] for stream_id in stream_order]

                q.put({
                    "batches": torched_batch,
                    "stream_ids": stream_order,
                    "user_params": user_params,
                    "fps": fps,
                })

                for stream_id in stream_order:
                    pending_frames[stream_id] = None
    finally:
        for ctx in stream_contexts:
            ctx["cap"].release()

    stop_thread(q)