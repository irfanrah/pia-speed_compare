import json
import os
from queue import Empty, Queue

import cv2
import numpy as np
import pytest
import torch

from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.DTO.param_base import ROIModel, RetrievalBase
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.global_config import ALARMS_KEY
from pia_prod.AI.modules.ft_pe.config import (
    ALARM_QUEUE_SIZE,
    ALARM_THRESHOLD,
    DEVICE,
    FT_PE_ID,
    FT_PE_MODEL_ONNX_PATH,
    FT_PE_MODEL_PYTORCH_PATH,
    FT_PE_MODEL_TRT_PATH,
    FT_TEXT_FEATURES_PATH,
    MODE_PRESETS,
    FTPEMode,
)
from pia_prod.AI.modules.ft_pe.trt_export import export_trt_engine
from pia_prod.AI.modules.ft_pe.utils import draw_multi_category_status
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.utils.init import logger
from pia_prod.AI.utils.utils import AddStreamModel2dict


CATEGORY_VIDEO_MAP = {
    "fire": "test_FTPE_fire.mp4",
    "falldown": "test_FTPE_falldown.mp4",
    "smoke": "test_FTPE_smoke.mp4",
    "violence": "test_FTPE_violence.mp4",
    "firesmoke": "test_FTPE_firesmoke.mp4",
}


@pytest.fixture
def setup_model_and_video(hf_downloader, nas_downloader, video_save_dir):
    """
    ft_pe 모듈 자체에서 필요한 자원만 준비한다 (pe_violence에 의존하지 않음).
    - HuggingFace: FT_PE .pt / .onnx / FT_text_features.json 다운로드
    - TRT 엔진: ft_pe.trt_export.export_trt_engine로 빌드
    - NAS: test_FTPE_<category>.mp4 4종 다운로드
    """
    pt_name = os.path.basename(FT_PE_MODEL_PYTORCH_PATH)
    onnx_name = os.path.basename(FT_PE_MODEL_ONNX_PATH)
    hf_downloader.download(repo_id=FT_PE_ID, file_name=pt_name, snapshot=False)
    hf_downloader.download(repo_id=FT_PE_ID, file_name=onnx_name, snapshot=False)
    hf_downloader.download(
        repo_id=FT_PE_ID,
        file_name=os.path.basename(FT_TEXT_FEATURES_PATH),
        snapshot=False,
    )

    export_trt_engine(
        onnx_file=FT_PE_MODEL_ONNX_PATH,
        save_file_path=FT_PE_MODEL_TRT_PATH,
        min_batch_size=1,
        opt_batch_size=4,
        max_batch_size=8,
    )

    ftpe_videos = {}
    for category, video_name in CATEGORY_VIDEO_MAP.items():
        local = os.path.join(video_save_dir, video_name)
        nas_downloader.download_file(nas_downloader.get_nas_path(video_name), local)
        ftpe_videos[category] = local

    assert os.path.exists(FT_TEXT_FEATURES_PATH), (
        f"FT_text_features.json not found at {FT_TEXT_FEATURES_PATH}"
    )

    return {
        "ftpe_videos": ftpe_videos,
        "video_save_dir": video_save_dir,
    }


def _make_user_param(camera_id, camera_url, organization, category="violence"):
    """
    해당 category에 대응하는 <category>_ft_ret 하나만 retEvent에 등록.
    camera "7"은 ROI 없음, 그 외는 샘플 사각형 ROI.
    """
    ret_event_name = f"{category}_ft_ret"
    if camera_id == "7":
        polygon_coordinates = []
    else:
        polygon_coordinates = [120, 90, 1820, 95, 1790, 1070, 110, 1050]
    return {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId=camera_id,
                cameraUrl=camera_url,
                organization=organization,
                retEvent=[
                    RetrievalBase(
                        name=ret_event_name,
                        incidentThresholdSecond=3,
                        incidentTimeoutSecond=3,
                        roi=ROIModel(roiId=1, polygonCoordinates=polygon_coordinates),
                        topCandidates=1,
                        abnormalText=[category],
                        normalText=["normal"],
                    ),
                ],
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }


WS1_PRESET = pytest.param(FTPEMode.FPS_8, id=FTPEMode.FPS_8.value)
WS3_PRESET = pytest.param(FTPEMode.FPS_3, id=FTPEMode.FPS_3.value)
ALL_PRESETS = [WS1_PRESET, WS3_PRESET]


def _apply_mode(monkeypatch, mode: FTPEMode):
    """
    FT_PE_MODE 프리셋 (WINDOW_SIZE, SLIDING_WINDOW_SIZE, PREDICTION_SIZE, TIME_INTERVAL)을
    ft_pe.service 모듈에 주입하고, 테스트에 필요한 input_fps도 함께 돌려준다.
    """
    from pia_prod.AI.modules.ft_pe import service as ft_service

    window_size, sliding_window_size, prediction_size, time_interval = MODE_PRESETS[mode]
    input_fps = int(round(1.0 / time_interval))
    monkeypatch.setattr(ft_service, "WINDOW_SIZE", window_size)
    monkeypatch.setattr(ft_service, "SLIDING_WINDOW_SIZE", sliding_window_size)
    monkeypatch.setattr(ft_service, "PREDICTION_SIZE", prediction_size)
    return ft_service, window_size, sliding_window_size, prediction_size, input_fps


def _apply_ft_json_txt_vectors(monkeypatch, category):
    """
    FT_text_features.json의 해당 category 블록 (normal + <category> text_features)을
    FTPEService.category_txt_vectors로 직접 주입한다.
    """
    from pia_prod.AI.modules.ft_pe import service as ft_service

    with open(FT_TEXT_FEATURES_PATH) as f:
        data = json.load(f)
    assert category in data, f"'{category}' block missing in {FT_TEXT_FEATURES_PATH}"
    block = data[category]
    assert "normal" in block["text_features"], (
        f"'normal' text_features missing for '{category}'"
    )
    assert category in block["text_features"], (
        f"'{category}' text_features missing for '{category}'"
    )

    def _load_text_vectors(self):
        self.category_txt_vectors = {}
        self.category_normal_vectors = {}
        normal_feats = block["text_features"]["normal"]
        abn_feats = block["text_features"][category]
        n = torch.tensor(normal_feats, dtype=torch.float32, device=DEVICE)
        n = n / n.norm(dim=-1, keepdim=True)
        self.category_normal_vectors[category] = n.t().contiguous()
        a = torch.tensor(abn_feats, dtype=torch.float32, device=DEVICE)
        a = a / a.norm(dim=-1, keepdim=True)
        self.category_txt_vectors[category] = a.t().contiguous()

    monkeypatch.setattr(ft_service.FTPEService, "_load_text_vectors", _load_text_vectors)


@pytest.mark.parametrize("mode", [WS1_PRESET])
def test_ft_pe_violence_single_camera(setup_model_and_video, monkeypatch, mode):
    """단일 스트림에 대해 WS1 프리셋으로 FTPEService가 스레드 추론 로직을 정상 수행하는지 확인."""
    (
        ft_service,
        window_size,
        sliding_window_size,
        prediction_size,
        input_fps,
    ) = _apply_mode(monkeypatch, mode)
    _apply_ft_json_txt_vectors(monkeypatch, "violence")

    video_path = setup_model_and_video["ftpe_videos"]["violence"]
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't Open the Video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps / input_fps))
    q = Queue(1)

    user_param = _make_user_param(
        camera_id="6", camera_url="11", organization="pia", category="violence"
    )
    stream_id = (
        f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    )

    service = ft_service.FTPEService(q)
    assert service.window_size == window_size
    assert service.sliding_window_size == sliding_window_size
    assert service.prediction_size == prediction_size

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

    cap.release()
    stop_thread(q)


@pytest.mark.parametrize("mode", [WS1_PRESET])
def test_ft_pe_violence_multiple_cameras(setup_model_and_video, monkeypatch, mode):
    """ROI 유/무 두 스트림이 WS1 프리셋에서 동작하는지 확인."""
    ft_service, _, _, _, input_fps = _apply_mode(monkeypatch, mode)
    _apply_ft_json_txt_vectors(monkeypatch, "violence")

    video_path_1 = setup_model_and_video["ftpe_videos"]["violence"]
    video_path_2 = setup_model_and_video["ftpe_videos"]["fire"]
    q = Queue(1)

    camera_specs = [
        {
            "video_path": video_path_1,
            "user_param": _make_user_param(
                camera_id="6", camera_url="11", organization="pia", category="violence"
            ),
        },
        {
            "video_path": video_path_2,
            "user_param": _make_user_param(
                camera_id="7", camera_url="12", organization="pia", category="violence"
            ),
        },
    ]

    probe_cap = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe_cap.isOpened(), "Can't Open the Video"
        fps = int(probe_cap.get(cv2.CAP_PROP_FPS))
    finally:
        probe_cap.release()

    interval = max(1, round(fps / input_fps))
    ft_service.FTPEService(q)

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
    pending_frames = {sid: None for sid in stream_order}

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

            if all(pending_frames[sid] is not None for sid in stream_order):
                frames = [pending_frames[sid] for sid in stream_order]
                user_params = [user_param_map[sid] for sid in stream_order]
                q.put(
                    {
                        "batches": frames,
                        "stream_ids": stream_order,
                        "user_params": user_params,
                        "fps": fps,
                    }
                )
                for sid in stream_order:
                    pending_frames[sid] = None
    finally:
        for ctx in stream_contexts:
            ctx["cap"].release()

    stop_thread(q)


def test_import_module():
    """Test Importing Service Module."""
    from pia_prod.AI import FTPEService

    assert FTPEService is not None, "FTPEService module import failed"


def _run_deterministic_single_camera(
    *,
    ft_service,
    setup_data,
    category,
    preset_id,
    window_size,
    sliding_window_size,
    prediction_size,
    input_fps,
):
    """
    Deterministic single-thread single-camera runner for a given anomaly category.
    - Queue.put + manual q.get_nowait() (스레드 소비 우회)
    - 디버그 영상 {video_save_dir}/debug_ft_pe_{category}_{preset_id}.mp4
    - 최소 1건의 알람 전환 decision이 발생해야 테스트 통과.
    """
    video_path = setup_data["ftpe_videos"][category]
    video_save_dir = setup_data["video_save_dir"]
    debug_out_path = os.path.join(
        video_save_dir, f"debug_ft_pe_{category}_{preset_id}.mp4"
    )

    ret_event_id = f"{category}_ft_ret"
    status_text = "NORMAL"
    alarm_queue_count = 0
    alarm_queue_entries = []
    is_triggered = False

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps and fps > 0, f"Invalid FPS: {fps}"

    q = Queue(maxsize=1)
    service = ft_service.FTPEService(q)
    service.debug = True
    assert service.window_size == window_size
    assert service.sliding_window_size == sliding_window_size
    assert service.prediction_size == prediction_size
    assert category in service.category_txt_vectors
    assert category in service.category_normal_vectors

    interval = max(1, round(fps / input_fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_out_path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter: {debug_out_path}"

    user_param = _make_user_param(
        camera_id="7", camera_url="11", organization="pia", category=category
    )
    stream_id = (
        f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    )

    decisions = []

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

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        vis_frame = frame.copy()

        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8

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

            eq = service.alarm_event_manager.duration_queue.get(stream_id)
            cat_q = eq.get(ret_event_id) if eq else None
            if cat_q is not None:
                alarm_queue_entries = list(cat_q)
                alarm_queue_count = int(sum(cat_q))
                is_triggered = (
                    alarm_queue_count >= service.alarm_event_manager.alarm_threshold
                )

            if detect_out is not None:
                alarms = detect_out[ALARMS_KEY]
                decisions.append((frame_count, alarms))

                val = alarms.get(stream_id, [None, None])[0]
                if val is True:
                    status_text = category.upper()
                elif val is False:
                    status_text = "NORMAL"

                logger.info(
                    f"[{category}/{preset_id}] [frame={frame_count}] status={status_text} "
                    f"queue={alarm_queue_count}/{ALARM_QUEUE_SIZE} "
                    f"thr={ALARM_THRESHOLD} triggered={is_triggered}"
                )

        draw_multi_category_status(
            vis_frame,
            frame_count,
            width,
            height,
            [(category, alarm_queue_entries, is_triggered)],
            queue_size=ALARM_QUEUE_SIZE,
            threshold=ALARM_THRESHOLD,
        )
        writer.write(vis_frame)

    cap.release()
    writer.release()

    # Terminate AI inference thread and release GPU memory so that subsequent
    # parametrized deterministic tests don't accumulate TRT engine allocations.
    stop_thread(q)

    logger.info(f"[{category}/{preset_id}] Detected {len(decisions)} decisions.")
    assert len(decisions) > 0, (
        f"[{category}/{preset_id}] FTPEService never produced a decision (no anomaly detected)"
    )


def _run_deterministic_for_category(setup_data, monkeypatch, mode, category, preset_id):
    ft_service, window_size, sliding_window_size, prediction_size, input_fps = _apply_mode(
        monkeypatch, mode
    )
    _apply_ft_json_txt_vectors(monkeypatch, category)
    _run_deterministic_single_camera(
        ft_service=ft_service,
        setup_data=setup_data,
        category=category,
        preset_id=preset_id,
        window_size=window_size,
        sliding_window_size=sliding_window_size,
        prediction_size=prediction_size,
        input_fps=input_fps,
    )


@pytest.mark.parametrize("mode", ALL_PRESETS)
def test_deterministic_ft_pe_fire_single_camera(
    setup_model_and_video, monkeypatch, request, mode
):
    _run_deterministic_for_category(
        setup_model_and_video, monkeypatch, mode, "fire", request.node.callspec.id
    )


@pytest.mark.parametrize("mode", ALL_PRESETS)
def test_deterministic_ft_pe_falldown_single_camera(
    setup_model_and_video, monkeypatch, request, mode
):
    _run_deterministic_for_category(
        setup_model_and_video, monkeypatch, mode, "falldown", request.node.callspec.id
    )


@pytest.mark.parametrize("mode", ALL_PRESETS)
def test_deterministic_ft_pe_smoke_single_camera(
    setup_model_and_video, monkeypatch, request, mode
):
    _run_deterministic_for_category(
        setup_model_and_video, monkeypatch, mode, "smoke", request.node.callspec.id
    )


@pytest.mark.parametrize("mode", ALL_PRESETS)
def test_deterministic_ft_pe_violence_single_camera(
    setup_model_and_video, monkeypatch, request, mode
):
    _run_deterministic_for_category(
        setup_model_and_video, monkeypatch, mode, "violence", request.node.callspec.id
    )


@pytest.mark.parametrize("mode", [WS3_PRESET])
def test_deterministic_ft_pe_firesmoke_simultaneous_single_camera(
    setup_model_and_video, monkeypatch, request, mode
):
    """
    Register fire_ft_ret + smoke_ft_ret + violence_ft_ret on a single stream and
    verify against test_FTPE_firesmoke.mp4 that:
      - fire and smoke cross the alarm threshold in the same decision step
        at least once (simultaneous detection)
      - violence is never triggered over the entire video (specificity)

    Text vectors are loaded via the real _load_text_vectors (no monkeypatch),
    so all category embeddings from FT_text_features.json are used — this
    matches the actual multi-category runtime behavior.

    The debug overlay draws one row per category, stacked top-down.
    """
    ft_service, window_size, sliding_window_size, prediction_size, input_fps = _apply_mode(
        monkeypatch, mode
    )

    preset_id = request.node.callspec.id
    video_path = setup_model_and_video["ftpe_videos"]["firesmoke"]
    video_save_dir = setup_model_and_video["video_save_dir"]
    debug_out_path = os.path.join(
        video_save_dir, f"debug_ft_pe_firesmoke_{preset_id}.mp4"
    )

    expected_positive = ["fire", "smoke"]
    expected_negative = ["violence"]
    categories = expected_positive + expected_negative
    ret_event_ids = [f"{cat}_ft_ret" for cat in categories]
    positive_ret_ids = [f"{cat}_ft_ret" for cat in expected_positive]
    negative_ret_ids = [f"{cat}_ft_ret" for cat in expected_negative]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps and fps > 0, f"Invalid FPS: {fps}"

    q = Queue(maxsize=1)
    service = ft_service.FTPEService(q)
    service.debug = True
    assert service.window_size == window_size
    assert service.sliding_window_size == sliding_window_size
    assert service.prediction_size == prediction_size
    for cat in categories:
        assert cat in service.category_txt_vectors, (
            f"'{cat}' abnormal vectors missing — check FT_text_features.json"
        )
        assert cat in service.category_normal_vectors, (
            f"'{cat}' normal vectors missing — check FT_text_features.json"
        )

    interval = max(1, round(fps / input_fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_out_path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter: {debug_out_path}"

    user_param = {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId="7",
                cameraUrl="11",
                organization="pia",
                retEvent=[
                    RetrievalBase(
                        name=rid,
                        incidentThresholdSecond=3,
                        incidentTimeoutSecond=3,
                        roi=ROIModel(roiId=1, polygonCoordinates=[]),
                        topCandidates=1,
                        abnormalText=[cat],
                        normalText=["normal"],
                    )
                    for cat, rid in zip(categories, ret_event_ids)
                ],
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }
    stream_id = (
        f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
    )

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

    simultaneous_frames = []
    per_category_trigger_frames = {rid: [] for rid in ret_event_ids}
    latest_queue = {rid: [] for rid in ret_event_ids}
    latest_triggered = {rid: False for rid in ret_event_ids}

    frame_count = 0
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
            _process_one_from_queue()

            eq = service.alarm_event_manager.duration_queue.get(stream_id)
            if eq is not None:
                for rid in ret_event_ids:
                    cat_q = eq.get(rid)
                    if cat_q is None:
                        latest_queue[rid] = []
                        latest_triggered[rid] = False
                        continue
                    cq_list = list(cat_q)
                    is_over = int(sum(cq_list)) >= service.alarm_event_manager.alarm_threshold
                    latest_queue[rid] = cq_list
                    latest_triggered[rid] = is_over
                    if is_over:
                        per_category_trigger_frames[rid].append(frame_count)

            if all(latest_triggered[rid] for rid in positive_ret_ids):
                simultaneous_frames.append(frame_count)

            logger.info(
                f"[firesmoke/{preset_id}] [frame={frame_count}] "
                + " ".join(
                    f"{rid}={'ON' if latest_triggered[rid] else 'OFF'}"
                    for rid in ret_event_ids
                )
            )

        category_rows = [
            (cat, latest_queue[rid], latest_triggered[rid])
            for cat, rid in zip(categories, ret_event_ids)
        ]
        draw_multi_category_status(
            vis_frame,
            frame_count,
            width,
            height,
            category_rows,
            queue_size=ALARM_QUEUE_SIZE,
            threshold=ALARM_THRESHOLD,
        )
        writer.write(vis_frame)

    cap.release()
    writer.release()
    stop_thread(q)

    for rid in positive_ret_ids:
        assert per_category_trigger_frames[rid], (
            f"[firesmoke/{preset_id}] expected-positive '{rid}' never triggered "
            f"in {video_path}"
        )
    for rid in negative_ret_ids:
        assert not per_category_trigger_frames[rid], (
            f"[firesmoke/{preset_id}] expected-negative '{rid}' was triggered at "
            f"frames {per_category_trigger_frames[rid][:5]}... "
            f"(count={len(per_category_trigger_frames[rid])})"
        )
    assert simultaneous_frames, (
        f"[firesmoke/{preset_id}] fire and smoke were never triggered in the same "
        f"decision step (fire steps={len(per_category_trigger_frames['fire_ft_ret'])}, "
        f"smoke steps={len(per_category_trigger_frames['smoke_ft_ret'])})"
    )

    logger.info(
        f"[firesmoke/{preset_id}] simultaneous fire+smoke triggers: "
        f"{len(simultaneous_frames)} steps, first at frame {simultaneous_frames[0]}. "
        f"violence triggers: {len(per_category_trigger_frames['violence_ft_ret'])} (expected 0)"
    )
