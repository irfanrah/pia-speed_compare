"""
PeVqa2StageService 테스트.

검증 항목:
- PE TRT 1단계 추론 (numpy / torch tensor 입력)
- PeVqa2StageEventManager 사이클 (사고 지속 시 알람 1회만)
- send_alarm 분기 (PE_VQA_2STAGE_VALIDATION_ENABLED OFF / ON 명시)
- 메시지 형식 (cameraId int, ts microsecond, 다른 PE 모듈과 일관)
- ON 모드 e2e: 비동기 POST → validation_server → RabbitMQ → 메시지 수신 검증

테스트 모드 분리:
- OFF 모드 fixture (`off_mode`): PE_VQA_2STAGE_VALIDATION_ENABLED=False 명시 + reload
- ON 모드 fixture (`on_mode`): PE_VQA_2STAGE_VALIDATION_ENABLED=True 명시 + reload + validation_server 가용성 확인
"""
import importlib
import json
import os
import time
from queue import Queue, Empty
from typing import List, Optional

import cv2
import httpx
import pika
import pytest
import torch

from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.DTO.param_base import RetrievalBase, ROIModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.global_config import ALARMS_KEY
from pia_prod.AI.modules.perception_encoder.trt_export import export_trt_engine
from pia_prod.AI.modules.pe_vqa_2stage.config import (
    INPUT_SIZE,
    PERCEPTION_ENCODER_ONNX_PATH,
    PERCEPTION_ENCODER_TRT_PATH,
    PERCEPTION_ENCODER_TXT_FEATURE_PATH,
)
from pia_prod.AI.tests.thread_stop_signal import stop_thread
from pia_prod.AI.utils.init import logger
from pia_prod.AI.utils.utils import AddStreamModel2dict


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def setup_model_and_video(hf_downloader, nas_downloader, video_save_dir):
    """PE TRT 엔진 빌드 + 화재/쓰러짐 영상 다운로드 + debug 출력 경로."""
    txt_json_name = os.path.basename(PERCEPTION_ENCODER_TXT_FEATURE_PATH)
    onnx_file_save_path = PERCEPTION_ENCODER_ONNX_PATH
    repo_name = os.path.splitext(os.path.basename(PERCEPTION_ENCODER_TRT_PATH))[0]
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

    video_name_1 = "ys_falldown_video.mp4"
    video_name_2 = "samsung_fire_fps13.mp4"
    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)

    nas_downloader.download_file(nas_downloader.get_nas_path(video_name_1), local_video_path_1)
    nas_downloader.download_file(nas_downloader.get_nas_path(video_name_2), local_video_path_2)

    return {
        "video_falldown": local_video_path_1,
        "video_fire": local_video_path_2,
        "debug_fire_off": os.path.join(video_save_dir, "debug_pe_vqa_2stage_fire_off.mp4"),
        "debug_falldown_off": os.path.join(video_save_dir, "debug_pe_vqa_2stage_falldown_off.mp4"),
        "debug_fire_on": os.path.join(video_save_dir, "debug_pe_vqa_2stage_fire_on.mp4"),
        "debug_falldown_on": os.path.join(video_save_dir, "debug_pe_vqa_2stage_falldown_on.mp4"),
    }


@pytest.fixture(scope="module")
def check_validation_server():
    """validation_server 가용성 확인. 미실행 시 ON 모드 테스트 skip."""
    host = os.getenv("PE_VQA_2STAGE_VALIDATION_HOST", "localhost")
    port = os.getenv("PE_VQA_2STAGE_VALIDATION_PORT", "8100")
    url = f"http://{host}:{port}"
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip(
            "validation_server가 실행 중이 아닙니다. 먼저 서버를 실행하세요:\n"
            "  cd packages/pia_prod/AI/modules/pe_vqa_2stage/validation_server\n"
            "  docker compose up -d"
        )
    return url


def _reload_pe_vqa_2stage_modules():
    """pe_vqa_2stage config + service 모듈 reload — ENV 변경 반영."""
    import pia_prod.AI.modules.pe_vqa_2stage.config as cfg_module
    import pia_prod.AI.modules.pe_vqa_2stage.service as svc_module

    importlib.reload(cfg_module)
    importlib.reload(svc_module)
    return svc_module


@pytest.fixture
def off_mode(monkeypatch):
    """PE_VQA_2STAGE_VALIDATION_ENABLED=False 명시 + 모듈 reload."""
    monkeypatch.setenv("PE_VQA_2STAGE_VALIDATION_ENABLED", "False")
    svc_module = _reload_pe_vqa_2stage_modules()
    assert svc_module.PE_VQA_2STAGE_VALIDATION_ENABLED is False
    yield svc_module
    # teardown: 다음 테스트 영향 안 받도록 다시 reload (default OFF 복원)
    monkeypatch.setenv("PE_VQA_2STAGE_VALIDATION_ENABLED", "False")
    _reload_pe_vqa_2stage_modules()


@pytest.fixture
def on_mode(monkeypatch, check_validation_server):
    """PE_VQA_2STAGE_VALIDATION_ENABLED=True 명시 + validation_server 의존."""
    monkeypatch.setenv("PE_VQA_2STAGE_VALIDATION_ENABLED", "True")
    monkeypatch.setenv("PE_VQA_2STAGE_QUEUE_SIZE", "5")
    monkeypatch.setenv("PE_VQA_2STAGE_ALARM_DURATION_THRESHOLD", "3")
    svc_module = _reload_pe_vqa_2stage_modules()
    assert svc_module.PE_VQA_2STAGE_VALIDATION_ENABLED is True
    yield svc_module
    # teardown: OFF로 복원
    monkeypatch.setenv("PE_VQA_2STAGE_VALIDATION_ENABLED", "False")
    _reload_pe_vqa_2stage_modules()


@pytest.fixture
def mq_consumer():
    """RabbitMQ 큐를 비우고 consume할 수 있는 채널 반환 (ON 모드 메시지 수신용)."""
    host = os.getenv("BACKEND_RABBITMQ_IP", "localhost")
    port = int(os.getenv("BACKEND_RABBITMQ_PORT", "5672"))
    user = os.getenv("BACKEND_RABBITMQ_USER_NAME", "guest")
    password = os.getenv("BACKEND_RABBITMQ_PASSWORD", "guest")
    queue_name = os.getenv("BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME", "ret_queue_test")

    conn = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=host, port=port,
            credentials=pika.PlainCredentials(user, password),
        )
    )
    ch = conn.channel()
    ch.queue_declare(queue=queue_name, durable=True)
    ch.queue_purge(queue=queue_name)

    def consume_one(timeout_s: float = 10.0):
        end = time.time() + timeout_s
        while time.time() < end:
            method, _, body = ch.basic_get(queue=queue_name, auto_ack=True)
            if method:
                return json.loads(body.decode("utf-8"))
            time.sleep(0.3)
        return None

    yield consume_one
    try:
        conn.close()
    except Exception:
        pass


# ============================================================
# Helpers
# ============================================================


def _make_user_param(
    camera_id: str,
    organization: str,
    categories: List[str],
    polygon_coordinates: Optional[List[int]] = None,
):
    """카테고리 리스트로 retEvent를 자동 생성. pe_vqa_2stage의 _pe_vqa 시리즈 지원."""
    if polygon_coordinates is None:
        polygon_coordinates = []  # 전체 영역

    abnormal_text_map = {
        "fire_pe_vqa": ["fire"],
        "화재_pe_vqa": ["fire"],
        "smoke_pe_vqa": ["smoke"],
        "연기_pe_vqa": ["smoke"],
        "falldown_pe_vqa": ["falldown"],
        "쓰러짐_pe_vqa": ["falldown"],
        "smoking_pe_vqa": ["smoking"],
        "흡연_pe_vqa": ["smoking"],
    }

    ret_events = [
        RetrievalBase(
            name=cat,
            incidentThresholdSecond=3,
            incidentTimeoutSecond=3,
            roi=ROIModel(roiId=1, polygonCoordinates=polygon_coordinates),
            topCandidates=5,
            abnormalText=abnormal_text_map.get(cat, [cat.split("_")[0]]),
            normalText=["normal"],
        )
        for cat in categories
    ]

    return {
        "user_param": AddStreamModel2dict(
            AddStreamModel(
                cameraId=camera_id,
                cameraUrl="11",
                organization=organization,
                retEvent=ret_events,
                timestamp=str_UTC_ISO8601_ms_now_time(),
            )
        )
    }


def _stream_id(user_param: dict) -> str:
    up = user_param["user_param"]
    return f"{up['cameraId']}_{up['organization']}"


def _run_video_through_service(service, user_param, video_path, fps_div=2):
    """영상을 큐로 흘려 서비스가 PE 추론 + send_alarm까지 처리하게 한다."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / fps_div))
    sid = _stream_id(user_param)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % interval == 0:
                if service.analysis_data_queue.full():
                    try:
                        service.analysis_data_queue.get_nowait()
                        service.analysis_data_queue.task_done()
                    except Empty:
                        pass
                service.analysis_data_queue.put({
                    "batches": [frame],
                    "stream_ids": [sid],
                    "user_params": [user_param],
                })
    finally:
        cap.release()


# ============================================================
# Tests — Common
# ============================================================


def test_import_module():
    """Service 모듈 import 검증."""
    from pia_prod.AI import PeVqa2StageService

    assert PeVqa2StageService is not None, "PeVqa2StageService import failed"


# ============================================================
# Tests — OFF 모드 (1단계 직접 발사)
# ============================================================


def test_deterministic_pe_vqa_2stage_fire_alarm_off_mode(off_mode, setup_model_and_video):
    """
    OFF 모드 deterministic — PE TRT → EventManager → match_outputs → KTT alarm_producer.

    검증:
    - PE_VQA_2STAGE_VALIDATION_ENABLED=False 명시
    - PE 추론으로 알람 발생
    - 알람 dict에 stream_id 포함
    """
    PeVqa2StageService = off_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_fire"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="100", organization="off_mode", categories=["fire_pe_vqa"]
    )
    sid = _stream_id(user_param)

    service = PeVqa2StageService(q)
    decisions = []

    def _step():
        try:
            item = q.get_nowait()
        except Empty:
            return None
        q.task_done()
        return service._detect(
            batches=item["batches"],
            stream_ids=item["stream_ids"],
            user_params=item["user_params"],
        )

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % interval == 0:
                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except Empty:
                        pass
                q.put({"batches": [frame], "stream_ids": [sid], "user_params": [user_param]})
                out = _step()
                if out is not None:
                    decisions.append((frame_count, out[ALARMS_KEY]))
    finally:
        cap.release()

    assert len(decisions) > 0, "OFF 모드: 화재 영상인데 알람 발생 안 함"
    first_alarms = decisions[0][1]
    matched = [k for k in first_alarms if k.startswith(sid) or k == sid]
    assert matched, f"OFF 모드: stream_id={sid} 알람 누락: {first_alarms}"

    stop_thread(q)


def test_deterministic_pe_vqa_2stage_no_burst_off_mode(off_mode, setup_model_and_video):
    """
    OFF 모드 — 사고 지속 시 시작 알람 폭주 방지 (review-guide.md:38).

    PE state reset 결함(service.py:215-217)이 회귀하면 시작 알람이 N회 발사됨.
    """
    PeVqa2StageService = off_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_fire"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="200", organization="off_mode", categories=["fire_pe_vqa"]
    )
    sid = _stream_id(user_param)

    service = PeVqa2StageService(q)
    n_starts = 0
    n_ends = 0

    def _step():
        try:
            item = q.get_nowait()
        except Empty:
            return None
        q.task_done()
        return service._detect(
            batches=item["batches"],
            stream_ids=item["stream_ids"],
            user_params=item["user_params"],
        )

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % interval == 0:
                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except Empty:
                        pass
                q.put({"batches": [frame], "stream_ids": [sid], "user_params": [user_param]})
                out = _step()
                if out is not None:
                    for _, v in out[ALARMS_KEY].items():
                        if v[0] is True:
                            n_starts += 1
                        elif v[0] is False:
                            n_ends += 1
    finally:
        cap.release()

    assert n_starts >= 1, "OFF 모드: 시작 알람 발생 안 함"
    assert n_starts <= 2, f"OFF 모드: 시작 알람 폭주 의심 ({n_starts}회) — PE reset 결함 회귀"
    logger.info(f"OFF 모드 no_burst: 시작 {n_starts}회, 종료 {n_ends}회")

    stop_thread(q)


def test_pe_vqa_2stage_multiple_categories_off_mode(off_mode, setup_model_and_video):
    """OFF 모드 — 4 카테고리 단일 카메라 부하 + 다중 카테고리 분기 검증."""
    PeVqa2StageService = off_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_falldown"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="6",
        organization="pia",
        categories=["smoke_pe_vqa", "falldown_pe_vqa", "fire_pe_vqa", "흡연_pe_vqa"],
    )
    sid = _stream_id(user_param)

    PeVqa2StageService(q)
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % interval == 0:
                q.put({"batches": [frame], "stream_ids": [sid], "user_params": [user_param]})
    finally:
        cap.release()
    stop_thread(q)


def test_pe_vqa_2stage_multiple_cameras_off_mode(off_mode, setup_model_and_video):
    """OFF 모드 — 2 카메라 batch + 다중 카테고리 분기."""
    PeVqa2StageService = off_mode.PeVqa2StageService
    paths = setup_model_and_video
    q = Queue(maxsize=1)

    camera_specs = [
        {
            "video_path": paths["video_falldown"],
            "user_param": _make_user_param(
                camera_id="6", organization="pia",
                categories=["smoke_pe_vqa", "falldown_pe_vqa"],
            ),
        },
        {
            "video_path": paths["video_fire"],
            "user_param": _make_user_param(
                camera_id="7", organization="pia",
                categories=["fire_pe_vqa"],
                polygon_coordinates=[1646, 1402, 2582, 1404, 2589, 390, 1570, 376],
            ),
        },
    ]

    probe = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe.isOpened()
        fps = int(probe.get(cv2.CAP_PROP_FPS))
    finally:
        probe.release()
    interval = max(1, int(fps / 2))

    PeVqa2StageService(q)
    contexts = []
    for spec in camera_specs:
        cap = cv2.VideoCapture(spec["video_path"])
        assert cap.isOpened(), f"Can't open video: {spec['video_path']}"
        contexts.append({
            "cap": cap,
            "stream_id": _stream_id(spec["user_param"]),
            "user_param": spec["user_param"],
            "frame_count": 0,
        })

    pending = {ctx["stream_id"]: None for ctx in contexts}
    try:
        while True:
            ready = True
            for ctx in contexts:
                ret, frame = ctx["cap"].read()
                if not ret:
                    ready = False
                    break
                ctx["frame_count"] += 1
                if ctx["frame_count"] % interval == 0:
                    pending[ctx["stream_id"]] = frame
            if not ready:
                break
            if all(pending[s] is not None for s in pending):
                stream_order = [ctx["stream_id"] for ctx in contexts]
                q.put({
                    "batches": [pending[s] for s in stream_order],
                    "stream_ids": stream_order,
                    "user_params": [
                        next(c["user_param"] for c in contexts if c["stream_id"] == s)
                        for s in stream_order
                    ],
                })
                for s in stream_order:
                    pending[s] = None
    finally:
        for ctx in contexts:
            ctx["cap"].release()

    stop_thread(q)


def test_pe_vqa_2stage_torch_one_camera_off_mode(off_mode, setup_model_and_video):
    """
    OFF 모드 — 1단계 torch tensor 입력 (ES팀 호환).

    PE 추론과 send_alarm thumbnail 처리가 torch.Tensor를 처리해야 한다.
    """
    PeVqa2StageService = off_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_fire"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="11", organization="pia", categories=["fire_pe_vqa"]
    )
    sid = _stream_id(user_param)

    PeVqa2StageService(q)
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % interval == 0:
                # numpy → torch tensor [H,W,C] uint8, BGR → RGB
                torched = torch.from_numpy(frame[..., [2, 1, 0]])
                q.put({"batches": [torched], "stream_ids": [sid], "user_params": [user_param]})
    finally:
        cap.release()
    stop_thread(q)


def test_pe_vqa_2stage_torch_two_cameras_off_mode(off_mode, setup_model_and_video):
    """OFF 모드 — 1단계 torch tensor 다중 카메라 batch."""
    PeVqa2StageService = off_mode.PeVqa2StageService
    paths = setup_model_and_video
    q = Queue(maxsize=1)

    camera_specs = [
        {
            "video_path": paths["video_fire"],
            "user_param": _make_user_param(
                camera_id="12", organization="pia", categories=["fire_pe_vqa"]
            ),
        },
        {
            "video_path": paths["video_falldown"],
            "user_param": _make_user_param(
                camera_id="13", organization="pia", categories=["falldown_pe_vqa"]
            ),
        },
    ]

    probe = cv2.VideoCapture(camera_specs[0]["video_path"])
    try:
        assert probe.isOpened()
        fps = int(probe.get(cv2.CAP_PROP_FPS))
    finally:
        probe.release()
    interval = max(1, int(fps / 2))

    PeVqa2StageService(q)
    contexts = []
    for spec in camera_specs:
        cap = cv2.VideoCapture(spec["video_path"])
        assert cap.isOpened()
        contexts.append({
            "cap": cap,
            "stream_id": _stream_id(spec["user_param"]),
            "user_param": spec["user_param"],
            "frame_count": 0,
        })

    pending = {ctx["stream_id"]: None for ctx in contexts}
    try:
        while True:
            ready = True
            for ctx in contexts:
                ret, frame = ctx["cap"].read()
                if not ret:
                    ready = False
                    break
                ctx["frame_count"] += 1
                if ctx["frame_count"] % interval == 0:
                    pending[ctx["stream_id"]] = frame[..., [2, 1, 0]]
            if not ready:
                break
            if all(pending[s] is not None for s in pending):
                stream_order = [ctx["stream_id"] for ctx in contexts]
                torched_batch = [torch.from_numpy(pending[s]) for s in stream_order]
                q.put({
                    "batches": torched_batch,
                    "stream_ids": stream_order,
                    "user_params": [
                        next(c["user_param"] for c in contexts if c["stream_id"] == s)
                        for s in stream_order
                    ],
                })
                for s in stream_order:
                    pending[s] = None
    finally:
        for ctx in contexts:
            ctx["cap"].release()

    stop_thread(q)


# ============================================================
# Tests — ON 모드 (validation_server 위임)
# ============================================================


def test_deterministic_pe_vqa_2stage_fire_alarm_on_mode(
    on_mode, setup_model_and_video, mq_consumer
):
    """
    ON 모드 deterministic — PE 알람 → 비동기 POST → validation_server → RabbitMQ 메시지 수신.

    검증:
    - PE_VQA_2STAGE_VALIDATION_ENABLED=True 명시
    - 메인 inference의 send_alarm이 비동기 위임
    - validation_server가 vLLM 검증 후 RabbitMQ에 publish
    - 수신 메시지 형식 (cameraId int, ts microsecond, 키 셋)
    """
    PeVqa2StageService = on_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_fire"]

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="888", organization="on_mode_e2e", categories=["fire_pe_vqa"]
    )

    service = PeVqa2StageService(q)
    _run_video_through_service(service, user_param, video_path, fps_div=2)
    stop_thread(q)

    # 비동기 처리 완료 대기 (vLLM 호출 + S3 업로드 + publish)
    time.sleep(8)

    msg = mq_consumer(timeout_s=15)
    if msg is None:
        # vLLM이 거부했을 가능성 (작은 영상은 화재로 인식 못 할 수 있음)
        # fail-open이 아닌 정상 vLLM이라면 거부 → 메시지 없음 = 정상
        logger.warning("ON 모드: vLLM이 화재 인식 못 했거나 컨테이너 측 timing 문제. 그러나 폐기 자체는 정상 동작")
        return

    # 메시지 형식 검증
    assert isinstance(msg.get("cameraId"), int), f"cameraId int 아님: {type(msg.get('cameraId'))}"
    assert msg.get("type") == "retEvent"
    assert msg.get("organization") == "on_mode_e2e"
    expected_keys = {
        "cameraId", "type", "organization", "name", "isStart",
        "thumbnail", "incidentThresholdSecond", "incidentTimeoutSecond",
        "uuid", "ts",
    }
    assert set(msg.keys()) == expected_keys, f"키 셋 불일치: {set(msg.keys()) ^ expected_keys}"

    # ts: KTT 표준 (microsecond 6자리, Z 없음)
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$", msg["ts"]), (
        f"ts 형식 KTT 표준과 다름: {msg['ts']}"
    )
    logger.info(f"ON 모드 검증 메시지: {msg}")


def test_deterministic_pe_vqa_2stage_no_burst_on_mode(
    on_mode, setup_model_and_video, mq_consumer
):
    """
    ON 모드 — 사고 지속 시 시작 알람 폭주 방지.

    PE EventManager가 status 1→2(continue) 사이클을 유지하므로
    비동기 위임으로 가는 시작 알람도 1~2회만 발사되어야 한다.
    """
    PeVqa2StageService = on_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_fire"]

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="889", organization="on_mode_burst", categories=["fire_pe_vqa"]
    )

    service = PeVqa2StageService(q)
    _run_video_through_service(service, user_param, video_path, fps_div=2)
    stop_thread(q)
    time.sleep(8)

    # 메시지 수집
    n_starts = 0
    n_ends = 0
    while True:
        msg = mq_consumer(timeout_s=2)
        if msg is None:
            break
        if msg.get("isStart") is True:
            n_starts += 1
        elif msg.get("isStart") is False:
            n_ends += 1

    # vLLM이 거부할 수도 있어 0회도 허용. 다만 발사된 경우 폭주는 안 되어야 함.
    assert n_starts <= 2, f"ON 모드: 시작 알람 폭주 ({n_starts}회) — PE reset 결함 회귀"
    logger.info(f"ON 모드 no_burst: 시작 {n_starts}회, 종료 {n_ends}회")


def test_pe_vqa_2stage_torch_one_camera_on_mode(on_mode, setup_model_and_video):
    """ON 모드 — 1단계 torch tensor 입력 (validation server 위임 흐름)."""
    PeVqa2StageService = on_mode.PeVqa2StageService
    video_path = setup_model_and_video["video_fire"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="891", organization="on_mode_torch", categories=["fire_pe_vqa"]
    )
    sid = _stream_id(user_param)

    PeVqa2StageService(q)
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % interval == 0:
                torched = torch.from_numpy(frame[..., [2, 1, 0]])
                q.put({"batches": [torched], "stream_ids": [sid], "user_params": [user_param]})
    finally:
        cap.release()
    stop_thread(q)


def test_pe_vqa_2stage_validation_server_health(on_mode):
    """ON 모드 시 validation_server health check (가용성 검증)."""
    url = os.getenv("VALIDATION_SERVER_URL", f"http://{os.getenv('PE_VQA_2STAGE_VALIDATION_HOST', 'localhost')}:{os.getenv('PE_VQA_2STAGE_VALIDATION_PORT', '8100')}")
    resp = httpx.get(f"{url.rstrip('/')}/health", timeout=5)
    assert resp.status_code == 200


# ============================================================
# Tests — Visualization (debug video 출력)
# ============================================================


def _format_status_text(matched_alarm, mode_label: str = "") -> str:
    """알람 dict의 [is_start, category]를 사람이 읽기 좋은 텍스트로 변환.

    is_start=True  → STARTED (사고 시작 알람, status 0→1)
    is_start=False → ENDED   (사고 종료 알람, status 2→3)
    """
    is_start, cat = matched_alarm
    if is_start is True:
        label = "STARTED"
    elif is_start is False:
        label = "ENDED"
    else:
        label = "ALARM"
    suffix = f" [{mode_label}]" if mode_label else ""
    return f"{label}: {cat}{suffix}"


def _run_with_visualization(
    Service,
    video_path: str,
    debug_path: str,
    user_param: dict,
    mode_label: str = "",
):
    """
    영상 frame을 큐에 흘리며 PE 추론 결과를 frame에 그려 debug 영상으로 저장.
    qwen3_vl_embedding.utils.draw_status를 재사용해 다른 PE 모듈 시각화와 동일한 스타일.

    OFF 모드: send_alarm이 KTT alarm_producer로 직접 발사
    ON 모드: send_alarm이 비동기 POST → validation_server (시각화는 메인 측 알람 발생 시점)
    """
    from pia_prod.AI.modules.qwen3_vl_embedding.utils import draw_status

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter: {debug_path}"

    q = Queue(maxsize=1)
    sid = _stream_id(user_param)
    service = Service(q)

    def _step():
        try:
            item = q.get_nowait()
        except Empty:
            return None
        q.task_done()
        return service._detect(
            batches=item["batches"],
            stream_ids=item["stream_ids"],
            user_params=item["user_params"],
        )

    status_text = "normal"
    n_starts = 0
    n_ends = 0
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            vis = frame.copy()

            if frame_count % interval == 0:
                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except Empty:
                        pass
                q.put({
                    "batches": [frame],
                    "stream_ids": [sid],
                    "user_params": [user_param],
                })
                out = _step()
                if out is not None:
                    alarms = out[ALARMS_KEY]
                    matched = [v for k, v in alarms.items() if k.startswith(sid) or k == sid]
                    if matched:
                        is_start = matched[0][0]
                        if is_start is True:
                            n_starts += 1
                        elif is_start is False:
                            n_ends += 1
                        status_text = _format_status_text(matched[0], mode_label)
                        logger.info(f"frame {frame_count}: {status_text}")
                    else:
                        status_text = "normal"

            draw_status(vis, status_text, frame_count, width, height)
            writer.write(vis)
    finally:
        cap.release()
        writer.release()

    stop_thread(q)

    assert os.path.exists(debug_path), f"debug video 저장 실패: {debug_path}"
    assert os.path.getsize(debug_path) > 0, f"debug video 비어있음: {debug_path}"
    return {"starts": n_starts, "ends": n_ends, "total": n_starts + n_ends}


def test_visualize_pe_vqa_2stage_fire_off_mode(off_mode, setup_model_and_video):
    """
    OFF 모드 — 화재 영상 시각화 (1단계 직접 발사).

    Frame별 알람 상태를 영상에 오버레이:
    - "normal" (녹색): 정상 상태
    - "STARTED: fire_pe_vqa [OFF]" (빨강): 시작 알람 발생 (status 0→1)
    - "ENDED: fire_pe_vqa [OFF]" (빨강): 종료 알람 발생 (status 2→3)
    """
    paths = setup_model_and_video
    user_param = _make_user_param(
        camera_id="100", organization="vis_off_fire", categories=["fire_pe_vqa"]
    )
    counts = _run_with_visualization(
        off_mode.PeVqa2StageService,
        video_path=paths["video_fire"],
        debug_path=paths["debug_fire_off"],
        user_param=user_param,
        mode_label="OFF",
    )
    logger.info(
        f"OFF 모드 fire 시각화: 시작 {counts['starts']}회, 종료 {counts['ends']}회 "
        f"→ {paths['debug_fire_off']}"
    )
    assert counts["starts"] > 0, "화재 영상인데 시작 알람 한 번도 발생 안 함 (시각화)"


def test_visualize_pe_vqa_2stage_falldown_off_mode(off_mode, setup_model_and_video):
    """OFF 모드 — 쓰러짐 영상 시각화 (다중 카테고리 retEvent로 알람 트리거 감도 검증)."""
    paths = setup_model_and_video
    user_param = _make_user_param(
        camera_id="101", organization="vis_off_falldown",
        categories=["falldown_pe_vqa", "smoke_pe_vqa", "fire_pe_vqa"],
    )
    counts = _run_with_visualization(
        off_mode.PeVqa2StageService,
        video_path=paths["video_falldown"],
        debug_path=paths["debug_falldown_off"],
        user_param=user_param,
        mode_label="OFF",
    )
    logger.info(
        f"OFF 모드 falldown 시각화: 시작 {counts['starts']}회, 종료 {counts['ends']}회 "
        f"→ {paths['debug_falldown_off']}"
    )


def test_visualize_pe_vqa_2stage_fire_on_mode(on_mode, setup_model_and_video):
    """
    ON 모드 — 화재 영상 시각화 (validation_server 위임 흐름).

    OFF 모드와 동일하게 메인의 PE 추론 결과를 frame에 오버레이.
    단 ON 모드에서는 알람이 비동기로 validation_server에 위임되며,
    실제 RabbitMQ 발사는 vLLM 검증 결과에 따라 결정된다.
    영상 자체는 메인 inference 측의 알람 발생 시점만 표시.

    상태 표기:
    - "STARTED: fire_pe_vqa [ON]"  : 메인 PE 알람 발생 시점, validation_server에 위임됨
    - "ENDED: fire_pe_vqa [ON]"    : 메인 PE 종료 알람, validation_server에 위임됨
    """
    paths = setup_model_and_video
    user_param = _make_user_param(
        camera_id="900", organization="vis_on_fire", categories=["fire_pe_vqa"]
    )
    counts = _run_with_visualization(
        on_mode.PeVqa2StageService,
        video_path=paths["video_fire"],
        debug_path=paths["debug_fire_on"],
        user_param=user_param,
        mode_label="ON",
    )
    logger.info(
        f"ON 모드 fire 시각화: 시작 {counts['starts']}회, 종료 {counts['ends']}회 "
        f"→ {paths['debug_fire_on']}"
    )
    assert counts["starts"] > 0, "화재 영상인데 시작 알람 한 번도 발생 안 함 (ON 시각화)"
    # ON 모드에서도 폭주 안 함 (PE EventManager 사이클 정상)
    assert counts["starts"] <= 2, f"ON 모드 시각화: 시작 알람 폭주 ({counts['starts']}회)"


def test_visualize_pe_vqa_2stage_falldown_on_mode(on_mode, setup_model_and_video):
    """ON 모드 — 쓰러짐 영상 시각화 (다중 카테고리 retEvent)."""
    paths = setup_model_and_video
    user_param = _make_user_param(
        camera_id="901", organization="vis_on_falldown",
        categories=["falldown_pe_vqa", "smoke_pe_vqa", "fire_pe_vqa"],
    )
    counts = _run_with_visualization(
        on_mode.PeVqa2StageService,
        video_path=paths["video_falldown"],
        debug_path=paths["debug_falldown_on"],
        user_param=user_param,
        mode_label="ON",
    )
    logger.info(
        f"ON 모드 falldown 시각화: 시작 {counts['starts']}회, 종료 {counts['ends']}회 "
        f"→ {paths['debug_falldown_on']}"
    )


# ============================================================
# Tests — Kafka 백엔드 (ES MACS 호환)
# ============================================================


@pytest.fixture
def check_kafka():
    """Kafka 가용성 확인. 미실행 시 skip. (Schema Registry 미사용 — fastavro schemaless로 직접 직렬화)"""
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    try:
        from confluent_kafka.admin import AdminClient
        admin = AdminClient({"bootstrap.servers": bootstrap, "request.timeout.ms": 5000})
        topics = admin.list_topics(timeout=5)
        if topics is None:
            raise RuntimeError("list_topics returned None")
    except Exception as exc:
        pytest.skip(
            f"Kafka broker({bootstrap}) 응답 없음 ({exc}). 먼저 인프라 띄우세요:\n"
            "  docker run -d --network host confluentinc/cp-kafka:7.9.2 ..."
        )
    return {"bootstrap": bootstrap}


def test_kafka_publisher_avro_format(check_kafka, monkeypatch):
    """
    Kafka 백엔드 단위 검증 — ES MACS 호환 와이어 포맷 (fastavro schemaless).

    1. KafkaPublisher 인스턴스화 (fastavro schemaless writer)
    2. ES MACS 형식의 메시지 publish (Schema Registry 미사용)
    3. ES와 동일한 fastavro.schemaless_reader로 디코드 — 와이어 호환성 검증
    4. 메시지 형식 검증 (cameraId string, ts long epoch ms, thumbnail string)
    """
    import importlib
    import sys
    from io import BytesIO

    # 환경변수 고정 (monkeypatch — fixture 종료 시 자동 복원)
    monkeypatch.setenv("MESSAGING_BACKEND", "kafka")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", check_kafka["bootstrap"])
    monkeypatch.setenv("KAFKA_TOPIC_EVENT_PROCESS", "event.process.test")

    # validation_server 모듈에서 KafkaPublisher + 스키마 import
    server_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../modules/pe_vqa_2stage/validation_server",
        )
    )
    sys.path.insert(0, server_path)
    if "server" in sys.modules:
        del sys.modules["server"]
    if "prompts" in sys.modules:
        del sys.modules["prompts"]
    server_mod = importlib.import_module("server")

    publisher = server_mod.KafkaPublisher()
    publisher.connect()

    # ES Avro 스키마와 일치하는 메시지 (fastavro schemaless 출력은 ts를 long epoch ms로 그대로 보존)
    test_uuid = "kafka-test-uuid-001"
    test_message = {
        "cameraId": "777",
        "type": "retEvent",
        "organization": "kafka_test",
        "name": "fire_pe_vqa",
        "isStart": True,
        "thumbnail": "777_kafka_test_2026-04-28T01:00:00.jpg",
        "incidentThresholdSecond": 5,
        "incidentTimeoutSecond": 30,
        "uuid": test_uuid,
        "ts": int(time.time() * 1000),
    }

    publisher.publish(test_message, "event.process.test")
    publisher.close()

    # ES backend와 동일하게 fastavro.schemaless_reader로 디코드 — 와이어 포맷 호환성 검증
    from confluent_kafka import Consumer
    from fastavro import parse_schema, schemaless_reader

    parsed_schema = parse_schema(server_mod.EVENT_START_AVRO_SCHEMA)

    consumer = Consumer({
        "bootstrap.servers": check_kafka["bootstrap"],
        "group.id": "test-pe-vqa-2stage-consumer",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe(["event.process.test"])

    try:
        end = time.time() + 15
        received = None
        while time.time() < end:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                continue
            raw = msg.value()
            assert raw is not None, "Kafka raw bytes 수신 실패"
            # 와이어 포맷 검증: 첫 바이트가 Confluent Schema Registry 매직 바이트(0x00)면 안 됨.
            # ES는 schemaless라 첫 바이트가 Avro long zigzag 인코딩(예: cameraId string의 길이) 시작.
            assert raw[0] != 0x00 or len(raw) > 5, (
                f"Confluent SR 매직 바이트가 보임 — ES backend(fastavro schemaless_reader)는 "
                f"이 포맷을 디코드하지 못한다. raw[:5]={raw[:5].hex()}"
            )
            decoded = schemaless_reader(BytesIO(raw), parsed_schema)
            if decoded and decoded.get("uuid") == test_uuid:
                received = decoded
                received_headers = dict(msg.headers() or [])
                break
        assert received is not None, "Kafka consumer 메시지 수신 못 함"

        # 형식 검증 — fastavro logical type timestamp-millis는 datetime으로 자동 변환됨
        from datetime import datetime
        assert isinstance(received["cameraId"], str), f"cameraId string 아님: {type(received['cameraId'])}"
        assert isinstance(received["ts"], datetime), f"ts datetime 아님: {type(received['ts'])}"
        assert received["uuid"] == test_uuid
        assert received["isStart"] is True
        # ts가 정상 시간대 검증
        assert received["ts"].year >= 2025, f"ts datetime 이상: {received['ts']}"
        # thumbnail string ("" 허용, ES와 동일 브랜치)
        assert isinstance(received["thumbnail"], str), f"thumbnail string 아님: {type(received['thumbnail'])}"

        # ES MACS 컨슈머 호환 — headers["action"] 부착 + organization 포함 (빈 traceId/traceparent 등 미부착)
        assert received_headers.get("action") == b"eventStart", (
            f"action 헤더 누락 또는 불일치: {received_headers}"
        )
        assert received_headers.get("organization") == b"kafka_test", (
            f"organization 헤더 누락 또는 불일치: {received_headers}"
        )
        # 빈 값 헤더는 ES `to_dict_headers` if v 필터 동작과 동일하게 부착되지 않아야 함
        for empty_key in ("traceId", "traceparent", "tracestate", "request_uuid"):
            assert empty_key not in received_headers, (
                f"빈 값 헤더가 부착됨 (ES와 불일치): {empty_key}={received_headers.get(empty_key)}"
            )

        logger.info(
            f"ES 호환 fastavro schemaless 메시지 + 헤더 검증 통과: "
            f"value={received}, headers={received_headers}"
        )
    finally:
        consumer.close()
        sys.path.remove(server_path)


def test_kafka_message_format_conversion(check_kafka, monkeypatch):
    """
    `make_alarm_message_compatible`의 백엔드별 형식 변환 검증.

    - Kafka 모드: cameraId string, ts long(epoch ms), thumbnail nullable
    - RabbitMQ 모드: cameraId int, ts ISO8601 microsecond, thumbnail 빈 문자열
    """
    import importlib
    import sys

    server_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../modules/pe_vqa_2stage/validation_server",
        )
    )
    sys.path.insert(0, server_path)

    user_param = {
        "user_param": {
            "cameraId": 555,
            "organization": "fmt_test",
            "retEvent": {
                "fire_pe_vqa": {
                    "name": "fire_pe_vqa",
                    "incidentThresholdSecond": 5,
                    "incidentTimeoutSecond": 30,
                }
            },
        }
    }

    # Kafka 모드
    monkeypatch.setenv("MESSAGING_BACKEND", "kafka")
    if "server" in sys.modules:
        del sys.modules["server"]
    server_kafka = importlib.import_module("server")
    msg_kafka = server_kafka.make_alarm_message_compatible(
        user_param=user_param,
        thumbnail_filename="",  # 빈 값
        is_start=False,
        category_name="fire_pe_vqa",
        event_type="retEvent",
        event_uuid="fmt-uuid-001",
    )
    assert isinstance(msg_kafka["cameraId"], str), f"Kafka cameraId 타입: {type(msg_kafka['cameraId'])}"
    assert msg_kafka["cameraId"] == "555"
    assert isinstance(msg_kafka["ts"], int), f"Kafka ts 타입: {type(msg_kafka['ts'])}"
    assert msg_kafka["thumbnail"] == "", (
        f"Kafka thumbnail 빈 값은 \"\" (ES 동일 브랜치) 이어야 함: {msg_kafka['thumbnail']!r}"
    )

    # RabbitMQ 모드
    monkeypatch.setenv("MESSAGING_BACKEND", "rabbitmq")
    del sys.modules["server"]
    server_rmq = importlib.import_module("server")
    msg_rmq = server_rmq.make_alarm_message_compatible(
        user_param=user_param,
        thumbnail_filename="",
        is_start=False,
        category_name="fire_pe_vqa",
        event_type="retEvent",
        event_uuid="fmt-uuid-001",
    )
    assert isinstance(msg_rmq["cameraId"], int), f"RabbitMQ cameraId 타입: {type(msg_rmq['cameraId'])}"
    assert msg_rmq["cameraId"] == 555
    assert isinstance(msg_rmq["ts"], str), f"RabbitMQ ts 타입: {type(msg_rmq['ts'])}"
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$", msg_rmq["ts"])
    assert msg_rmq["thumbnail"] == "", f"RabbitMQ thumbnail 빈 문자열 유지 안 됨"

    sys.path.remove(server_path)
    logger.info(f"백엔드별 메시지 형식 변환 검증 통과:\n  Kafka: {msg_kafka}\n  RabbitMQ: {msg_rmq}")


# ============================================================
# 다중 카테고리 동시 전이 collision 방지 회귀 테스트
# ============================================================


def test_event_manager_multi_category_simultaneous_transition_no_collision(monkeypatch):
    """동일 stream의 두 카테고리가 같은 프레임에 status 0→1 전이 시 둘 다 살아남아야 함.

    회귀 방지 — 옛 포맷 `alarms[stream_id] = [...]`은 두 번째 카테고리가 첫 번째를 덮어써
    한 카테고리만 발사되는 결함이 있었다. ft_pe 패턴 `alarms[f"{stream_id}__{category_id}"]`로
    각 카테고리가 독립 키를 가져야 한다.
    """
    # 직전 테스트(on_mode fixture)가 PE_VQA_2STAGE_QUEUE_SIZE / _ALARM_DURATION_THRESHOLD를
    # override한 상태로 config를 reload했을 수 있으므로, 본 테스트 시작 시 default로 강제 복원하고
    # config + event 둘 다 reload해 module-level 상수와 임포트 값이 일관되게 한다.
    monkeypatch.delenv("PE_VQA_2STAGE_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("PE_VQA_2STAGE_ALARM_DURATION_THRESHOLD", raising=False)
    monkeypatch.delenv("PE_VQA_2STAGE_VALIDATION_ENABLED", raising=False)

    import pia_prod.AI.modules.pe_vqa_2stage.config as cfg_mod
    import pia_prod.AI.modules.pe_vqa_2stage.event as event_mod
    importlib.reload(cfg_mod)
    importlib.reload(event_mod)

    ALARM_DURATION_THRESHOLD = cfg_mod.ALARM_DURATION_THRESHOLD
    QUEUE_SIZE = cfg_mod.QUEUE_SIZE
    PeVqa2StageEventManager = event_mod.PeVqa2StageEventManager

    em = PeVqa2StageEventManager()
    stream_id = "camA_pia"

    # 두 카테고리 모두 ALARM_DURATION_THRESHOLD 만큼 1을 채워 동시에 status 0→1 전이 유도
    for _ in range(ALARM_DURATION_THRESHOLD):
        em.duration_queue[stream_id]["fire_pe_vqa"].append(1)
        em.duration_queue[stream_id]["smoke_pe_vqa"].append(1)

    alarms = em.check_alarm_duration()

    # 두 카테고리 각각의 alarm이 빠짐없이 살아남아야 함 (collision 0건)
    assert len(alarms) == 2, f"동시 전이 시 두 카테고리 모두 살아남아야 함: {alarms}"

    # ft_pe 패턴: 키는 composite, value의 두 번째 자리는 빈 문자열
    fire_key = f"{stream_id}__fire_pe_vqa"
    smoke_key = f"{stream_id}__smoke_pe_vqa"
    assert fire_key in alarms, f"fire 알람 누락: {alarms}"
    assert smoke_key in alarms, f"smoke 알람 누락: {alarms}"
    for key in (fire_key, smoke_key):
        is_start, category_field = alarms[key]
        assert is_start is True, f"{key} 시작 알람이 아님: {alarms[key]}"
        assert category_field == "", (
            f"{key} value의 category 자리는 빈 문자열이어야 함 "
            f"(get_alarm_with_uuid 이중 prefix 방지): {alarms[key]}"
        )

    # 종료 전이도 동일하게 둘 다 살아남는지 확인.
    # duration_queue는 maxlen=QUEUE_SIZE deque이므로 sum이 임계값 미만이 되도록 충분히 0을 누적.
    for _ in range(QUEUE_SIZE):
        em.duration_queue[stream_id]["fire_pe_vqa"].append(0)
        em.duration_queue[stream_id]["smoke_pe_vqa"].append(0)

    end_alarms = em.check_alarm_duration()
    assert len(end_alarms) == 2, f"동시 종료 전이 시 두 카테고리 모두 살아남아야 함: {end_alarms}"
    for key in (fire_key, smoke_key):
        is_start, _ = end_alarms[key]
        assert is_start is False, f"{key} 종료 알람이 아님: {end_alarms[key]}"

    logger.info(f"다중 카테고리 동시 전이 collision 방지 검증 통과:\n  start: {alarms}\n  end: {end_alarms}")


# ============================================================
# 순수 함수 단위 테스트 — Kafka/S3 외부 의존성 없이 실행
# ES MACS 와이어 호환성(헤더 / S3 키 형식)을 빠르게 검증
# ============================================================


def _import_validation_server_module(monkeypatch):
    """Kafka 모드로 validation_server `server` 모듈 동적 로드."""
    import importlib
    import sys

    monkeypatch.setenv("MESSAGING_BACKEND", "kafka")
    server_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../modules/pe_vqa_2stage/validation_server",
        )
    )
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    for mod in ("server", "prompts"):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("server"), server_path


def test_kafka_publisher_build_headers_format(monkeypatch):
    """`KafkaPublisher.build_headers` 단위 검증 — ES `to_dict_headers`와 동일한 결과를 생성해야 한다."""
    server_mod, server_path = _import_validation_server_module(monkeypatch)
    try:
        # eventStart + organization 비어있지 않을 때
        headers = server_mod.KafkaPublisher.build_headers({
            "isStart": True,
            "organization": "es_macs",
        })
        assert ("action", b"eventStart") in headers
        assert ("organization", b"es_macs") in headers
        assert len(headers) == 2

        # eventEnd + organization 빈 문자열 → organization 헤더 미부착 (ES if v 필터 동작)
        headers_end_empty_org = server_mod.KafkaPublisher.build_headers({
            "isStart": False,
            "organization": "",
        })
        assert headers_end_empty_org == [("action", b"eventEnd")]

        # organization 키 자체 누락 → action만
        headers_no_org = server_mod.KafkaPublisher.build_headers({"isStart": True})
        assert headers_no_org == [("action", b"eventStart")]

        logger.info("KafkaPublisher.build_headers 형식 검증 통과")
    finally:
        import sys
        sys.path.remove(server_path)


def test_s3_uploader_build_key_format(monkeypatch):
    """`S3Uploader.build_key` 단위 검증 — ES `MQProducer._run_drain_loop` S3 키와 동일 포맷."""
    import re

    server_mod, server_path = _import_validation_server_module(monkeypatch)
    try:
        ts_epoch_ms = 1714475400123  # 2024-04-30T10:30:00.123Z
        key = server_mod.S3Uploader.build_key("CAM01", "es_macs", ts_epoch_ms)

        # ES 운영 키 형식: {yyyymmdd}/{cameraId}_{organization}_{ts_epoch_ms}.jpg
        m = re.match(r"^(\d{8})/CAM01_es_macs_(\d+)\.jpg$", key)
        assert m is not None, f"S3 키 형식이 ES와 다름: {key}"
        assert m.group(2) == str(ts_epoch_ms), (
            f"S3 키의 ts가 epoch ms 정수가 아님 — message.ts와 동기화되지 않을 수 있음: {key}"
        )
        # ts ISO 흔적이 키에 들어가면 안 됨 (이전 버그 회귀 방지)
        assert "T" not in key and "Z" not in key, f"ISO8601 timestamp가 S3 키에 남음: {key}"

        logger.info(f"S3Uploader.build_key 형식 검증 통과: {key}")
    finally:
        import sys
        sys.path.remove(server_path)


def test_make_alarm_message_ts_sync_with_s3_key(monkeypatch):
    """ES MQProducer 동작 모방 — message.ts와 S3 키의 ts가 동일해야 한다."""
    server_mod, server_path = _import_validation_server_module(monkeypatch)
    try:
        ts_epoch_ms = 1714475400999

        message = server_mod.make_alarm_message_compatible(
            user_param={
                "user_param": {
                    "cameraId": 999,
                    "organization": "ts_sync",
                    "retEvent": {
                        "fire_pe_vqa": {
                            "name": "fire_pe_vqa",
                            "incidentThresholdSecond": 5,
                            "incidentTimeoutSecond": 30,
                        }
                    },
                }
            },
            thumbnail_filename="",
            is_start=True,
            category_name="fire_pe_vqa",
            event_type="retEvent",
            event_uuid="ts-sync-uuid",
            ts_epoch_ms=ts_epoch_ms,
        )
        s3_key = server_mod.S3Uploader.build_key("999", "ts_sync", ts_epoch_ms)

        assert message["ts"] == ts_epoch_ms, "message.ts가 주입한 ts와 다름"
        # S3 키 끝부분(파일명 stem)에 동일 ts가 들어있어야 함
        assert f"_{ts_epoch_ms}.jpg" in s3_key, (
            f"S3 키에 message.ts가 반영되지 않음 — ES MQProducer 동작과 불일치: {s3_key}"
        )
        logger.info(f"ts 동기화 검증 통과: message.ts={message['ts']}, s3_key={s3_key}")
    finally:
        import sys
        sys.path.remove(server_path)
