"""
PeVle2StageAsyncService tests (pe_vle_2stage_async).

Coverage:
- PE TRT Stage-1 inference (numpy / torch tensor input).
- PeVleEventManager cycle (one alarm per ongoing incident, no burst).
- send_alarm branching (PE_VLE_VALIDATION_ENABLED OFF / ON).
- Message format (cameraId int, ts microsecond — consistent with other PE modules).
- ON-mode e2e: async POST → validation_server (Qwen3VLE in-process)
                → RabbitMQ → received-message format check.

Mode separation (mirrors pe_vqa_2stage):
- OFF-mode fixture (`off_mode`): PE_VLE_VALIDATION_ENABLED=False + module reload.
- ON-mode fixture (`on_mode`): PE_VLE_VALIDATION_ENABLED=True + module reload
                                + validation_server reachability check.
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
from pia_prod.AI.modules.pe_vle_2stage_async.config import (
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
    """Build the PE TRT engine + download fire/falldown videos + reserve debug output paths."""
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

    video_name_1 = "qwen3vle_fire.mp4"
    video_name_2 = "two_stage_pe_qwen3vle_normal.mp4"
    local_video_path_1 = os.path.join(video_save_dir, video_name_1)
    local_video_path_2 = os.path.join(video_save_dir, video_name_2)

    nas_downloader.download_file(nas_downloader.get_nas_path(video_name_1), local_video_path_1)
    nas_downloader.download_file(nas_downloader.get_nas_path(video_name_2), local_video_path_2)

    return {
        # Anomaly clip (fire content) and normal clip — same pair the sync test uses.
        "video_anomaly": local_video_path_1,
        "video_normal":  local_video_path_2,
        "debug_anomaly_off": os.path.join(video_save_dir, "debug_pe_vle_async_anomaly_off.mp4"),
        "debug_normal_off":  os.path.join(video_save_dir, "debug_pe_vle_async_normal_off.mp4"),
        "debug_anomaly_on":  os.path.join(video_save_dir, "debug_pe_vle_async_anomaly_on.mp4"),
        "debug_normal_on":   os.path.join(video_save_dir, "debug_pe_vle_async_normal_on.mp4"),
        "debug_anomaly_deterministic": os.path.join(video_save_dir, "debug_pe_vle_async_anomaly_deterministic.mp4"),
        "debug_normal_deterministic":  os.path.join(video_save_dir, "debug_pe_vle_async_normal_deterministic.mp4"),
    }


@pytest.fixture(scope="module")
def check_validation_server():
    """Skip ON-mode tests if validation_server isn't reachable.

    pe_vle's validation_server loads Qwen3VLE in-process, so cold-start takes
    ~60-120s. A 200 from /health means both the anchor JSON and the vLLM
    weights have finished loading.
    """
    host = os.getenv("PE_VLE_VALIDATION_HOST", "localhost")
    port = os.getenv("PE_VLE_VALIDATION_PORT", "8200")
    url = f"http://{host}:{port}"
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip(
            "validation_server is not running. Start it first:\n"
            "  cd packages/pia_prod/AI/modules/pe_vle_2stage_async/validation_server\n"
            "  docker compose up -d\n"
            "  # cold-start ~60-120s — wait until /health returns 200"
        )
    return url


def _reload_pe_vle_modules():
    """Reload pe_vle_2stage_async config + service modules to pick up ENV changes."""
    import pia_prod.AI.modules.pe_vle_2stage_async.config as cfg_module
    import pia_prod.AI.modules.pe_vle_2stage_async.service as svc_module

    importlib.reload(cfg_module)
    importlib.reload(svc_module)
    return svc_module


@pytest.fixture
def off_mode(monkeypatch):
    """Force PE_VLE_VALIDATION_ENABLED=False + reload modules."""
    monkeypatch.setenv("PE_VLE_VALIDATION_ENABLED", "False")
    svc_module = _reload_pe_vle_modules()
    assert svc_module.PE_VLE_VALIDATION_ENABLED is False
    yield svc_module
    monkeypatch.setenv("PE_VLE_VALIDATION_ENABLED", "False")
    _reload_pe_vle_modules()


@pytest.fixture
def on_mode(monkeypatch, check_validation_server):
    """Force PE_VLE_VALIDATION_ENABLED=True + depend on a reachable validation_server."""
    monkeypatch.setenv("PE_VLE_VALIDATION_ENABLED", "True")
    monkeypatch.setenv("PE_VLE_QUEUE_SIZE", "5")
    monkeypatch.setenv("PE_VLE_ALARM_DURATION_THRESHOLD", "3")
    svc_module = _reload_pe_vle_modules()
    assert svc_module.PE_VLE_VALIDATION_ENABLED is True
    yield svc_module
    monkeypatch.setenv("PE_VLE_VALIDATION_ENABLED", "False")
    _reload_pe_vle_modules()


# Mirror of validation_server/publishers.py:EVENT_START_AVRO_SCHEMA — the
# Kafka publisher uses fastavro schemaless writes against this schema, so the
# test consumer must use the identical schema for schemaless_reader to decode.
# Keep in sync if the publisher's schema changes.
_EVENT_START_AVRO_SCHEMA = {
    "type": "record",
    "name": "eventStart",
    "namespace": "com.macs.events",
    "fields": [
        {"name": "cameraId", "type": "string"},
        {"name": "type", "type": "string"},
        {"name": "organization", "type": "string"},
        {"name": "name", "type": "string"},
        {"name": "isStart", "type": "boolean"},
        {"name": "thumbnail", "type": ["null", "string"], "default": None},
        {"name": "incidentThresholdSecond", "type": "int"},
        {"name": "incidentTimeoutSecond", "type": "int"},
        {"name": "uuid", "type": "string"},
        {"name": "ts", "type": {"type": "long", "logicalType": "timestamp-millis"}},
    ],
}


def _resolve_test_backend() -> str:
    """Pick the backend the test fixture should consume from.

    Priority: TEST_MESSAGING_BACKEND > MESSAGING_BACKEND > 'rabbitmq'.
    The dedicated TEST_ env lets the host-side test target a different backend
    than the validator container's own MESSAGING_BACKEND, since they're
    separate processes — but in the common case where the host inherits the
    same env, MESSAGING_BACKEND alone works.
    """
    backend = (
        os.getenv("TEST_MESSAGING_BACKEND")
        or os.getenv("MESSAGING_BACKEND")
        or "kafka"
    ).lower()
    if backend not in ("rabbitmq", "kafka"):
        raise ValueError(
            f"TEST_MESSAGING_BACKEND/MESSAGING_BACKEND={backend!r} "
            f"(must be 'rabbitmq' or 'kafka')"
        )
    return backend


@pytest.fixture
def mq_consumer():
    """Return a `consume_one(timeout_s)` callable bound to the active backend.

    Backend is resolved from TEST_MESSAGING_BACKEND / MESSAGING_BACKEND
    (default `rabbitmq`). Both branches yield decoded dicts in the
    backend-native shape — the test bodies adapt format assertions via
    `_resolve_test_backend()`.

    - rabbitmq: pika BlockingConnection on
      BACKEND_RABBITMQ_* env, JSON-decoded body. Queue is purged on entry so
      stale alarms from previous runs don't leak in.
    - kafka: confluent_kafka.Consumer on KAFKA_BOOTSTRAP_SERVERS, fastavro
      schemaless decode against `_EVENT_START_AVRO_SCHEMA`. Uses an ephemeral
      group_id and seeks to end-of-log on subscribe to mimic queue_purge —
      only messages produced *after* fixture setup are visible.
    """
    backend = _resolve_test_backend()

    if backend == "kafka":
        from confluent_kafka import Consumer, TopicPartition
        from fastavro import parse_schema, schemaless_reader
        from io import BytesIO

        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        topic = os.getenv("KAFKA_TOPIC_EVENT_PROCESS", "event.process")
        parsed_schema = parse_schema(_EVENT_START_AVRO_SCHEMA)

        # Manual assign() (vs subscribe()) avoids JoinGroup/FindCoordinator,
        # which routes through the broker's inter-broker listener and can hit
        # advertised-hostname resolution issues from a host-side test runner.
        # Tests don't need group offsets or rebalance — they just want to
        # tail messages produced after fixture setup.
        consumer = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": f"pe-vle-test-{os.getpid()}-{time.time_ns()}",
            "enable.auto.commit": False,
        })
        md = consumer.list_topics(topic, timeout=10.0)
        if topic not in md.topics or md.topics[topic].error is not None:
            consumer.close()
            raise RuntimeError(
                f"kafka topic {topic!r} not found via {bootstrap!r}: "
                f"{md.topics.get(topic).error if topic in md.topics else 'absent'}"
            )
        partitions = []
        for p in md.topics[topic].partitions.keys():
            tp = TopicPartition(topic, p)
            _, hi = consumer.get_watermark_offsets(tp, timeout=5.0)
            partitions.append(TopicPartition(topic, p, hi))
        consumer.assign(partitions)

        def consume_one(timeout_s: float = 10.0):
            msg = consumer.poll(timeout=timeout_s)
            if msg is None or msg.error():
                return None
            return schemaless_reader(BytesIO(msg.value()), parsed_schema)

        yield consume_one
        try:
            consumer.close()
        except Exception:
            pass
        return

    # rabbitmq branch
    host = os.getenv("BACKEND_RABBITMQ_IP", "localhost")
    port = int(os.getenv("BACKEND_RABBITMQ_PORT", "5672"))
    user = os.getenv("BACKEND_RABBITMQ_USER_NAME", "guest")
    password = os.getenv("BACKEND_RABBITMQ_PASSWORD", "guest")
    queue_name = os.getenv("BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME", "ret_queue_dev")

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
    """Build retEvent from a category list. Covers pe_vle_2stage_async's `_pe_vle_ret` series."""
    if polygon_coordinates is None:
        polygon_coordinates = []  # whole-frame ROI

    # Korean category strings ("화재_…", "연기_…", "쓰러짐_…") are the actual
    # labels shipped in pe_vle_2stage_async/config.py — leave them as data.
    abnormal_text_map = {
        "fire_pe_vle_ret":     ["fire"],
        "화재_pe_vle_ret":     ["fire"],
        "smoke_pe_vle_ret":    ["smoke"],
        "연기_pe_vle_ret":     ["smoke"],
        "falldown_pe_vle_ret": ["falldown"],
        "쓰러짐_pe_vle_ret":   ["falldown"],
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
    """Stream video frames into the service queue so the inference thread runs the
    full PE detection + send_alarm path."""
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
# Helpers — RabbitMQ
# ============================================================


def _drain_mq_messages(consume_one, per_call_timeout: float = 2.0) -> list:
    """Pull every queued message currently available and return them as a list.

    Works for both backends — the rabbit branch returns None when the queue
    drains, the kafka branch returns None when poll times out without a record.
    """
    messages = []
    while True:
        msg = consume_one(timeout_s=per_call_timeout)
        if msg is None:
            break
        messages.append(msg)
    return messages


def _assert_alarm_format(msg: dict, backend: str, expected_organization: str) -> None:
    """Backend-aware format check on a single isStart alarm.

    Wire shapes diverge by design (see validation_server/publishers.py
    `make_alarm_message_compatible`):
      rabbitmq (KTT):   cameraId int,  ts ISO8601 microsecond,  thumbnail str|None
      kafka (ES MACS):  cameraId str,  ts epoch-millis long,    thumbnail str ("" if empty)
    The non-format fields (type, organization, key set) are identical.
    """
    expected_keys = {
        "cameraId", "type", "organization", "name", "isStart",
        "thumbnail", "incidentThresholdSecond", "incidentTimeoutSecond",
        "uuid", "ts",
    }
    assert set(msg.keys()) == expected_keys, f"key-set mismatch: {set(msg.keys()) ^ expected_keys}"
    assert msg.get("type") == "retEvent"
    assert msg.get("organization") == expected_organization

    if backend == "kafka":
        import re
        assert isinstance(msg.get("cameraId"), str), (
            f"kafka avro: cameraId must be str, got {type(msg.get('cameraId'))}"
        )
        # Avro reads timestamp-millis as datetime by default; we want the long
        # epoch-ms form. fastavro returns datetime when logicalType is
        # honored — accept both shapes for portability.
        from datetime import datetime
        ts = msg.get("ts")
        assert isinstance(ts, (int, datetime)), (
            f"kafka avro: ts must be int (epoch-ms) or datetime, got {type(ts)}"
        )
        # cameraId must be a digit-string per the publisher's str(up["cameraId"]).
        assert re.match(r"^\d+$", msg["cameraId"]), (
            f"kafka avro: cameraId not digit-string: {msg['cameraId']!r}"
        )
    else:
        import re
        assert isinstance(msg.get("cameraId"), int), (
            f"rabbit json: cameraId must be int, got {type(msg.get('cameraId'))}"
        )
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$", msg["ts"]), (
            f"rabbit json: ts format diverges from KTT standard: {msg['ts']}"
        )


# ============================================================
# Component Tests — Import Sanity
# ============================================================


def test_import_module():
    """Sanity check that the service is importable from pia_prod.AI."""
    from pia_prod.AI import PeVle2StageAsyncService

    assert PeVle2StageAsyncService is not None, "PeVle2StageAsyncService import failed"


# ============================================================
# Component Tests — Publisher Message Formatting
# ============================================================


def _import_publishers_module(monkeypatch, backend: str = "kafka"):
    """Dynamically load the validation_server `publishers` + `config` modules."""
    import sys

    monkeypatch.setenv("MESSAGING_BACKEND", backend)
    server_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../modules/pe_vle_2stage_async/validation_server",
        )
    )
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    for mod in ("publishers", "config"):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("publishers"), server_path


def test_pe_vle_async_kafka_publisher_build_headers_format(monkeypatch):
    """`KafkaPublisher.build_headers` — must match ES `to_dict_headers` output shape."""
    publishers_mod, server_path = _import_publishers_module(monkeypatch)
    try:
        # eventStart + non-empty organization
        headers = publishers_mod.KafkaPublisher.build_headers({
            "isStart": True,
            "organization": "es_macs",
        })
        assert ("action", b"eventStart") in headers
        assert ("organization", b"es_macs") in headers
        assert len(headers) == 2

        # eventEnd + empty organization → organization header omitted
        headers_end_empty_org = publishers_mod.KafkaPublisher.build_headers({
            "isStart": False,
            "organization": "",
        })
        assert headers_end_empty_org == [("action", b"eventEnd")]

        # organization key absent → only action
        headers_no_org = publishers_mod.KafkaPublisher.build_headers({"isStart": True})
        assert headers_no_org == [("action", b"eventStart")]

        logger.info("KafkaPublisher.build_headers format check passed")
    finally:
        import sys
        sys.path.remove(server_path)


def test_pe_vle_async_s3_uploader_build_key_format(monkeypatch):
    """`S3Uploader.build_key` — must match ES `MQProducer._run_drain_loop` S3 key shape."""
    import re

    publishers_mod, server_path = _import_publishers_module(monkeypatch)
    try:
        ts_epoch_ms = 1714475400123  # 2024-04-30T10:30:00.123Z
        key = publishers_mod.S3Uploader.build_key("CAM01", "es_macs", ts_epoch_ms)

        m = re.match(r"^(\d{8})/CAM01_es_macs_(\d+)\.jpg$", key)
        assert m is not None, f"S3 key shape diverges from ES: {key}"
        assert m.group(2) == str(ts_epoch_ms), (
            f"S3 key ts is not an integer epoch ms: {key}"
        )
        assert "T" not in key and "Z" not in key, f"ISO8601 timestamp leaked into S3 key: {key}"

        logger.info(f"S3Uploader.build_key format check passed: {key}")
    finally:
        import sys
        sys.path.remove(server_path)


def test_pe_vle_async_make_alarm_message_ts_sync_with_s3_key(monkeypatch):
    """ES MQProducer parity — message.ts must equal the ts embedded in the S3 key."""
    publishers_mod, server_path = _import_publishers_module(monkeypatch)
    try:
        ts_epoch_ms = 1714475400999

        message = publishers_mod.make_alarm_message_compatible(
            user_param={
                "user_param": {
                    "cameraId": 999,
                    "organization": "ts_sync",
                    "retEvent": {
                        "fire_pe_vle_ret": {
                            "name": "fire_pe_vle_ret",
                            "incidentThresholdSecond": 5,
                            "incidentTimeoutSecond": 30,
                        }
                    },
                }
            },
            thumbnail_filename="",
            is_start=True,
            category_name="fire_pe_vle_ret",
            event_type="retEvent",
            event_uuid="ts-sync-uuid",
            ts_epoch_ms=ts_epoch_ms,
        )
        s3_key = publishers_mod.S3Uploader.build_key("999", "ts_sync", ts_epoch_ms)

        assert message["ts"] == ts_epoch_ms, "message.ts diverges from injected ts"
        assert f"_{ts_epoch_ms}.jpg" in s3_key, (
            f"S3 key does not embed message.ts: {s3_key}"
        )
        logger.info(f"ts sync check passed: message.ts={message['ts']}, s3_key={s3_key}")
    finally:
        import sys
        sys.path.remove(server_path)


def test_pe_vle_async_kafka_message_format_conversion(monkeypatch):
    """`make_alarm_message_compatible` — backend-aware message shape.

    - Kafka mode: cameraId string, ts long (epoch ms), thumbnail nullable.
    - RabbitMQ mode: cameraId int, ts ISO8601 microsecond, thumbnail empty string.
    """
    import re
    import sys

    user_param = {
        "user_param": {
            "cameraId": 555,
            "organization": "fmt_test",
            "retEvent": {
                "fire_pe_vle_ret": {
                    "name": "fire_pe_vle_ret",
                    "incidentThresholdSecond": 5,
                    "incidentTimeoutSecond": 30,
                }
            },
        }
    }

    # Kafka mode
    publishers_kafka, server_path = _import_publishers_module(monkeypatch, backend="kafka")
    msg_kafka = publishers_kafka.make_alarm_message_compatible(
        user_param=user_param,
        thumbnail_filename="",
        is_start=False,
        category_name="fire_pe_vle_ret",
        event_type="retEvent",
        event_uuid="fmt-uuid-001",
    )
    assert isinstance(msg_kafka["cameraId"], str), f"Kafka cameraId type: {type(msg_kafka['cameraId'])}"
    assert msg_kafka["cameraId"] == "555"
    assert isinstance(msg_kafka["ts"], int), f"Kafka ts type: {type(msg_kafka['ts'])}"
    assert msg_kafka["thumbnail"] == "", (
        f"Kafka thumbnail must be \"\" when empty (ES branch): {msg_kafka['thumbnail']!r}"
    )

    # RabbitMQ mode
    for mod in ("publishers", "config"):
        if mod in sys.modules:
            del sys.modules[mod]
    monkeypatch.setenv("MESSAGING_BACKEND", "rabbitmq")
    publishers_rmq = importlib.import_module("publishers")
    msg_rmq = publishers_rmq.make_alarm_message_compatible(
        user_param=user_param,
        thumbnail_filename="",
        is_start=False,
        category_name="fire_pe_vle_ret",
        event_type="retEvent",
        event_uuid="fmt-uuid-001",
    )
    assert isinstance(msg_rmq["cameraId"], int), f"RabbitMQ cameraId type: {type(msg_rmq['cameraId'])}"
    assert msg_rmq["cameraId"] == 555
    assert isinstance(msg_rmq["ts"], str), f"RabbitMQ ts type: {type(msg_rmq['ts'])}"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$", msg_rmq["ts"])
    assert msg_rmq["thumbnail"] == "", "RabbitMQ thumbnail did not preserve empty string"

    sys.path.remove(server_path)
    logger.info(
        f"Backend-aware message format check passed:\n  Kafka: {msg_kafka}\n  RabbitMQ: {msg_rmq}"
    )


# ============================================================
# Component Tests — Event Manager Regression
# ============================================================


def test_pe_vle_async_event_manager_multi_category_simultaneous_transition_no_collision(monkeypatch):
    """Two categories on the same stream transitioning 0→1 in the same frame must
    both survive in the alarms dict.

    Regression guard — the old `alarms[stream_id] = [...]` shape would have the
    second category overwrite the first, leaving only one fired. The composite-key
    pattern `alarms[f"{stream_id}__{category_id}"]` keeps each category independent.
    """
    monkeypatch.delenv("PE_VLE_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("PE_VLE_ALARM_DURATION_THRESHOLD", raising=False)
    monkeypatch.delenv("PE_VLE_VALIDATION_ENABLED", raising=False)

    import pia_prod.AI.modules.pe_vle_2stage_async.config as cfg_mod
    import pia_prod.AI.modules.pe_vle_2stage_async.event as event_mod
    importlib.reload(cfg_mod)
    importlib.reload(event_mod)

    ALARM_DURATION_THRESHOLD = cfg_mod.ALARM_DURATION_THRESHOLD
    QUEUE_SIZE = cfg_mod.QUEUE_SIZE
    PeVleEventManager = event_mod.PeVleEventManager

    em = PeVleEventManager()
    stream_id = "camA_pia"

    for _ in range(ALARM_DURATION_THRESHOLD):
        em.duration_queue[stream_id]["fire_pe_vle_ret"].append(1)
        em.duration_queue[stream_id]["smoke_pe_vle_ret"].append(1)

    alarms = em.check_alarm_duration()

    assert len(alarms) == 2, f"both categories must survive a simultaneous start: {alarms}"

    fire_key = f"{stream_id}__fire_pe_vle_ret"
    smoke_key = f"{stream_id}__smoke_pe_vle_ret"
    assert fire_key in alarms, f"fire alarm missing: {alarms}"
    assert smoke_key in alarms, f"smoke alarm missing: {alarms}"
    for key in (fire_key, smoke_key):
        is_start, category_field = alarms[key]
        assert is_start is True, f"{key} is not a start alarm: {alarms[key]}"
        assert category_field == "", (
            f"{key} value must have an empty category slot: {alarms[key]}"
        )

    # Verify the same survival on simultaneous end transitions.
    for _ in range(QUEUE_SIZE):
        em.duration_queue[stream_id]["fire_pe_vle_ret"].append(0)
        em.duration_queue[stream_id]["smoke_pe_vle_ret"].append(0)

    end_alarms = em.check_alarm_duration()
    assert len(end_alarms) == 2, f"both categories must survive a simultaneous end: {end_alarms}"
    for key in (fire_key, smoke_key):
        is_start, _ = end_alarms[key]
        assert is_start is False, f"{key} is not an end alarm: {end_alarms[key]}"

    logger.info(
        f"multi-category simultaneous-transition collision regression passed:\n"
        f"  start: {alarms}\n  end: {end_alarms}"
    )


# ============================================================
# Component Tests — Validation Server Internals
# ============================================================


def _import_server_module(monkeypatch):
    """Dynamically load validation_server `server` for AnomalyClassifier.

    The vLLM import remains lazy inside Qwen3VLEEmbedder.load(), so importing
    server.py does not require a GPU or model load.
    """
    import sys

    server_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../modules/pe_vle_2stage_async/validation_server",
        )
    )
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    monkeypatch.setenv("MESSAGING_BACKEND", "rabbitmq")
    for mod in ("server", "publishers", "config"):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("server"), server_path


def test_pe_vle_async_anomaly_classifier_target_vs_normal(monkeypatch, tmp_path):
    """AnomalyClassifier — confirms only when target_max > normal_max.

    Synthesizes a tiny anchor JSON to verify the cosine decision boundary.
    """
    # 4-D unit vectors so the cosine result is exact — anchor: normal in e0
    # direction, target in e1 direction. The real anchors are D=2048 but the
    # decision logic only needs orthogonal directions.
    fake_features = {
        "fire": {
            "text_features": {
                "normal": [[1.0, 0.0, 0.0, 0.0]],   # unit vector e0
                "fire":   [[0.0, 1.0, 0.0, 0.0]],   # unit vector e1
            }
        }
    }
    fake_path = tmp_path / "fake_text_features.json"
    fake_path.write_text(json.dumps(fake_features))

    monkeypatch.setenv("QWEN3VLE_VLLM_TEXT_FEATURES_PATH", str(fake_path))
    monkeypatch.setenv(
        "VLE_CATEGORY_EVENT_MAP_JSON",
        json.dumps({"fire": ["fire_vle_ret"]}),
    )

    server_mod, server_path = _import_server_module(monkeypatch)
    try:
        clf = server_mod.AnomalyClassifier()
        clf.load()
        assert "fire" in clf.anchors, f"fire bucket not loaded: {clf.anchors.keys()}"

        # Embedding aligned with the target direction → confirm.
        target_like = [0.0, 1.0, 0.0, 0.0]
        assert clf.classify(target_like, "fire_vle_ret") is True, (
            "target-aligned embedding was not confirmed"
        )

        # Embedding aligned with the normal direction → reject.
        normal_like = [1.0, 0.0, 0.0, 0.0]
        assert clf.classify(normal_like, "fire_vle_ret") is False, (
            "normal-aligned embedding was not rejected"
        )

        # Unmapped vle_id → fall-through (PE verdict trusted → True).
        assert clf.classify(target_like, "unknown_vle_ret") is True, (
            "unmapped vle_id should fall through to True"
        )

        logger.info("AnomalyClassifier decision-boundary check passed")
    finally:
        import sys
        sys.path.remove(server_path)


@pytest.fixture
def check_redis():
    """Skip Redis-dependent tests if no Redis is reachable."""
    import redis as _redis
    host = os.getenv("REDIS_IP", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    try:
        client = _redis.Redis(host=host, port=port, db=db, decode_responses=True)
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis({host}:{port}) unreachable ({exc})")
    return {"host": host, "port": port, "db": db}


def test_pe_vle_async_redis_uuid_tracker_pairing(check_redis, monkeypatch):
    """RedisUUIDTracker — add/contains/remove cycle.

    pe_vle's validation_server registers is_start UUIDs and looks them up when
    is_end arrives. If this cycle breaks, orphan ends slip through.
    """
    monkeypatch.setenv("UUID_KEY_PREFIX", "pe_vle_2stage_test:uuid:")
    monkeypatch.setenv("UUID_TTL_SECONDS", "60")

    publishers_mod, server_path = _import_publishers_module(monkeypatch)
    try:
        tracker = publishers_mod.RedisUUIDTracker(ttl_seconds=60)
        tracker.connect()

        test_uuid = f"pe-vle-async-tracker-test-{int(time.time())}"

        # Initial state: not registered.
        assert tracker.contains(test_uuid) is False, "fresh UUID reported as contains=True"

        # add → contains True.
        tracker.add(test_uuid)
        assert tracker.contains(test_uuid) is True, "contains=False after add"

        # remove → contains False.
        tracker.remove(test_uuid)
        assert tracker.contains(test_uuid) is False, "contains=True after remove"

        logger.info(f"RedisUUIDTracker pairing cycle passed: uuid={test_uuid}")
    finally:
        import sys
        sys.path.remove(server_path)


# --- Kafka e2e (optional — only when a Kafka broker is up) -------------------


@pytest.fixture
def check_kafka():
    """Skip Kafka e2e tests if no broker is reachable."""
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    try:
        from confluent_kafka.admin import AdminClient
        admin = AdminClient({"bootstrap.servers": bootstrap, "request.timeout.ms": 5000})
        topics = admin.list_topics(timeout=5)
        if topics is None:
            raise RuntimeError("list_topics returned None")
    except Exception as exc:
        pytest.skip(f"Kafka broker({bootstrap}) unreachable ({exc})")
    return {"bootstrap": bootstrap}


def test_pe_vle_async_kafka_publisher_avro_format(check_kafka, monkeypatch):
    """Kafka backend integration — fastavro schemaless wire format.

    1. Instantiate KafkaPublisher (fastavro schemaless writer).
    2. Publish an ES MACS-shaped message (no Schema Registry).
    3. Decode with fastavro.schemaless_reader (same call ES backend uses) →
       verifies wire compatibility.
    """
    import sys
    from datetime import datetime
    from io import BytesIO

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", check_kafka["bootstrap"])
    monkeypatch.setenv("KAFKA_TOPIC_EVENT_PROCESS", "event.process.pe_vle_test")

    publishers_mod, server_path = _import_publishers_module(monkeypatch, backend="kafka")
    try:
        publisher = publishers_mod.KafkaPublisher()
        publisher.connect()

        test_uuid = "pe-vle-kafka-test-uuid-001"
        test_message = {
            "cameraId": "777",
            "type": "retEvent",
            "organization": "pe_vle_kafka_test",
            "name": "fire_pe_vle_ret",
            "isStart": True,
            "thumbnail": "777_pe_vle_test_2026-04-28T01:00:00.jpg",
            "incidentThresholdSecond": 5,
            "incidentTimeoutSecond": 30,
            "uuid": test_uuid,
            "ts": int(time.time() * 1000),
        }

        publisher.publish(test_message, "event.process.pe_vle_test")
        publisher.close()

        from confluent_kafka import Consumer, TopicPartition
        from fastavro import parse_schema, schemaless_reader

        parsed_schema = parse_schema(publishers_mod.EVENT_START_AVRO_SCHEMA)
        consumer = Consumer({
            "bootstrap.servers": check_kafka["bootstrap"],
            "group.id": f"test-pe-vle-async-consumer-{os.getpid()}-{time.time_ns()}",
            "enable.auto.commit": False,
        })
        # Manual assign() (vs subscribe()) skips JoinGroup/FindCoordinator,
        # which routes through the broker's inter-broker listener and breaks
        # when the host can't resolve the advertised inter-broker hostname.
        # Read from low watermark so we can find our just-produced message.
        topic_name = "event.process.pe_vle_test"
        md = consumer.list_topics(topic_name, timeout=10.0)
        if topic_name not in md.topics or md.topics[topic_name].error is not None:
            consumer.close()
            pytest.fail(
                f"kafka topic {topic_name!r} not found via "
                f"{check_kafka['bootstrap']!r}: "
                f"{md.topics.get(topic_name).error if topic_name in md.topics else 'absent'}"
            )
        partitions = []
        for p in md.topics[topic_name].partitions.keys():
            tp = TopicPartition(topic_name, p)
            lo, _ = consumer.get_watermark_offsets(tp, timeout=5.0)
            partitions.append(TopicPartition(topic_name, p, lo))
        consumer.assign(partitions)

        try:
            end = time.time() + 15
            received = None
            received_headers = None
            while time.time() < end:
                msg = consumer.poll(timeout=1.0)
                if msg is None or msg.error():
                    continue
                raw = msg.value()
                assert raw is not None
                # A leading 0x00 byte is the Confluent SR magic byte — incompatible with ES.
                assert raw[0] != 0x00 or len(raw) > 5, (
                    f"Confluent SR magic byte detected — ES backend would fail to decode. "
                    f"raw[:5]={raw[:5].hex()}"
                )
                decoded = schemaless_reader(BytesIO(raw), parsed_schema)
                if decoded and decoded.get("uuid") == test_uuid:
                    received = decoded
                    received_headers = dict(msg.headers() or [])
                    break
            assert received is not None, "Kafka consumer never received a matching message"

            assert isinstance(received["cameraId"], str)
            assert isinstance(received["ts"], datetime)
            assert received["uuid"] == test_uuid
            assert received["isStart"] is True
            assert received["ts"].year >= 2025
            assert isinstance(received["thumbnail"], str)

            assert received_headers.get("action") == b"eventStart"
            assert received_headers.get("organization") == b"pe_vle_kafka_test"
            for empty_key in ("traceId", "traceparent", "tracestate", "request_uuid"):
                assert empty_key not in received_headers, (
                    f"empty-value header was attached: {empty_key}"
                )

            logger.info(
                f"ES-compatible fastavro schemaless message + headers check passed: "
                f"value={received}, headers={received_headers}"
            )
        finally:
            consumer.close()
    finally:
        sys.path.remove(server_path)


# ============================================================
# Integration Tests — Stage-1 / OFF Mode
# ============================================================


def test_deterministic_pe_vle_async_fire_alarm_off_mode(off_mode, setup_model_and_video):
    """OFF mode deterministic — PE TRT → EventManager → match_outputs → KTT alarm_producer."""
    PeVle2StageAsyncService = off_mode.PeVle2StageAsyncService
    video_path = setup_model_and_video["video_anomaly"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Can't open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="100", organization="off_mode", categories=["fire_pe_vle_ret"]
    )
    sid = _stream_id(user_param)

    service = PeVle2StageAsyncService(q)
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

    assert len(decisions) > 0, "OFF mode: no alarm fired on a fire video"
    first_alarms = decisions[0][1]
    matched = [k for k in first_alarms if k.startswith(sid) or k == sid]
    assert matched, f"OFF mode: alarm missing for stream_id={sid}: {first_alarms}"

    stop_thread(q)


def test_deterministic_pe_vle_async_no_burst_off_mode(off_mode, setup_model_and_video):
    """OFF mode — guard against start-alarm bursts during a sustained incident.

    A regression in PE state-reset would re-fire the start alarm N times.
    """
    PeVle2StageAsyncService = off_mode.PeVle2StageAsyncService
    video_path = setup_model_and_video["video_anomaly"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="200", organization="off_mode", categories=["fire_pe_vle_ret"]
    )
    sid = _stream_id(user_param)

    service = PeVle2StageAsyncService(q)
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

    assert n_starts >= 1, "OFF mode: no start alarm fired"
    assert n_starts <= 2, f"OFF mode: suspected start-alarm burst ({n_starts} starts)"
    logger.info(f"OFF mode no_burst: {n_starts} starts, {n_ends} ends")

    stop_thread(q)


def test_pe_vle_async_multiple_categories_off_mode(off_mode, setup_model_and_video):
    """OFF mode — single camera + 3 categories (load + multi-category branching)."""
    PeVle2StageAsyncService = off_mode.PeVle2StageAsyncService
    video_path = setup_model_and_video["video_anomaly"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="6",
        organization="pia",
        categories=["smoke_pe_vle_ret", "falldown_pe_vle_ret", "fire_pe_vle_ret"],
    )
    sid = _stream_id(user_param)

    PeVle2StageAsyncService(q)
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


def test_pe_vle_async_multiple_cameras_off_mode(off_mode, setup_model_and_video):
    """OFF mode — 2-camera batch with mixed categories + per-camera ROIs."""
    PeVle2StageAsyncService = off_mode.PeVle2StageAsyncService
    paths = setup_model_and_video
    q = Queue(maxsize=1)

    camera_specs = [
        {
            "video_path": paths["video_anomaly"],
            "user_param": _make_user_param(
                camera_id="6", organization="pia",
                categories=["fire_pe_vle_ret", "smoke_pe_vle_ret"],
            ),
        },
        {
            "video_path": paths["video_normal"],
            "user_param": _make_user_param(
                camera_id="7", organization="pia",
                categories=["fire_pe_vle_ret"],
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

    PeVle2StageAsyncService(q)
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


def test_pe_vle_async_torch_one_camera_off_mode(off_mode, setup_model_and_video):
    """OFF mode — Stage-1 with torch.Tensor input (ES-team interop)."""
    PeVle2StageAsyncService = off_mode.PeVle2StageAsyncService
    video_path = setup_model_and_video["video_anomaly"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="11", organization="pia", categories=["fire_pe_vle_ret"]
    )
    sid = _stream_id(user_param)

    PeVle2StageAsyncService(q)
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


def test_pe_vle_async_torch_two_cameras_off_mode(off_mode, setup_model_and_video):
    """OFF mode — Stage-1 torch.Tensor input across a multi-camera batch."""
    PeVle2StageAsyncService = off_mode.PeVle2StageAsyncService
    paths = setup_model_and_video
    q = Queue(maxsize=1)

    camera_specs = [
        {
            "video_path": paths["video_anomaly"],
            "user_param": _make_user_param(
                camera_id="12", organization="pia", categories=["fire_pe_vle_ret"]
            ),
        },
        {
            "video_path": paths["video_normal"],
            "user_param": _make_user_param(
                camera_id="13", organization="pia", categories=["fire_pe_vle_ret"]
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

    PeVle2StageAsyncService(q)
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
# Preflight Tests — Validation Stack
# ============================================================


def test_pe_vle_async_validation_server_health(on_mode):
    """ON mode — validation_server /health returns ok with anchor buckets loaded."""
    host = os.getenv("PE_VLE_VALIDATION_HOST", "localhost")
    port = os.getenv("PE_VLE_VALIDATION_PORT", "8200")
    url = f"http://{host}:{port}"
    resp = httpx.get(f"{url}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"
    # Anchor JSON must be loaded — lifespan would have raised RuntimeError otherwise.
    assert "anchor_buckets" in body, f"missing anchor_buckets in health response: {body}"
    assert isinstance(body["anchor_buckets"], list)
    logger.info(f"validation_server health: {body}")


def test_pe_vle_async_mq_preflight(mq_consumer):
    """Broker preflight — fixture must connect (rabbit or kafka) and start empty."""
    assert callable(mq_consumer)
    assert mq_consumer(timeout_s=0.1) is None


# ============================================================
# Main E2E Tests — ON Mode Anomaly / Normal
#
# Same scenario as the off-mode deterministic pair above, but exercises the
# full async pipeline:
#   PE → send_alarm → async POST → validation_server (vLLM embed +
#   AnomalyClassifier) → RabbitMQ publisher.
# Assertions look at messages received on the RabbitMQ queue instead of the
# direct _detect() output. Tests skip automatically when the validator or
# RabbitMQ aren't reachable.
# ============================================================


def test_deterministic_pe_vle_async_single_camera_anomaly_case_on_mode(
    on_mode, setup_model_and_video, mq_consumer,
):
    """ON-mode deterministic anomaly — fire video must surface as a published alarm.

    Verifies the full e2e: PE Stage-1 fires, validator embeds + confirms via
    AnomalyClassifier, the active broker (RabbitMQ KTT or Kafka ES MACS,
    selected via TEST_MESSAGING_BACKEND/MESSAGING_BACKEND) receives at least
    one isStart=True message. Format check is backend-aware — see
    `_assert_alarm_format`.
    """
    backend = _resolve_test_backend()
    PeVle2StageAsyncService = on_mode.PeVle2StageAsyncService
    paths = setup_model_and_video
    video_path = paths["video_anomaly"]

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="7", organization="on_anomaly_e2e",
        categories=["fire_pe_vle_ret", "smoke_pe_vle_ret"],
    )

    service = PeVle2StageAsyncService(q)
    _run_video_through_service(service, user_param, video_path, fps_div=2)
    stop_thread(q)

    # Wait for async work to drain (vLLM embed + S3 + Redis + publish).
    time.sleep(8)

    messages = _drain_mq_messages(mq_consumer, per_call_timeout=2.0)
    assert len(messages) > 0, (
        f"ON-mode anomaly ({backend}): fire video produced no messages. "
        "Either PE Stage-1 didn't fire or AnomalyClassifier rejected every start."
    )

    # At least one start alarm must have been published.
    starts = [m for m in messages if m.get("isStart") is True]
    assert len(starts) >= 1, (
        f"ON-mode anomaly ({backend}): no isStart=True message among "
        f"{len(messages)} received"
    )

    _assert_alarm_format(starts[0], backend, expected_organization="on_anomaly_e2e")

    logger.info(
        f"ON-mode anomaly e2e ({backend}): received {len(messages)} messages "
        f"({len(starts)} starts). First start: {starts[0]}"
    )


def test_deterministic_pe_vle_async_single_camera_normal_case_on_mode(
    on_mode, setup_model_and_video, mq_consumer,
):
    """ON-mode deterministic normal — normal video must publish nothing.

    Two paths can produce zero messages, both acceptable:
      1. PE Stage-1 never fires → send_alarm never runs → no POST to validator.
      2. PE Stage-1 fires a false positive → AnomalyClassifier rejects → no publish.
    The off-mode counterpart already asserts path (1); here we lock down the
    end-to-end "no message reaches the broker" guarantee (rabbit or kafka).
    """
    PeVle2StageAsyncService = on_mode.PeVle2StageAsyncService
    paths = setup_model_and_video
    video_path = paths["video_normal"]

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="7", organization="on_normal_e2e",
        categories=["fire_pe_vle_ret", "smoke_pe_vle_ret"],
    )

    service = PeVle2StageAsyncService(q)
    _run_video_through_service(service, user_param, video_path, fps_div=2)
    stop_thread(q)

    # Wait long enough for any spurious async POST to drain through the
    # validator and back-pressure on the queue to clear.
    time.sleep(8)

    messages = _drain_mq_messages(mq_consumer, per_call_timeout=2.0)
    if messages:
        # If anything published, surface what (helps diagnose: PE false positive
        # that AnomalyClassifier failed to reject, or a timing leak).
        logger.error(
            f"ON-mode normal: unexpected broker messages — {messages}"
        )
    assert len(messages) == 0, (
        f"ON-mode normal: normal video published {len(messages)} message(s) — "
        f"expected zero (PE Stage-1 false positive AnomalyClassifier failed to reject, "
        f"or PE_VLE_FAIL_OPEN is masking a vLLM error)."
    )

    logger.info("ON-mode normal e2e: no messages published, as expected.")


# ============================================================
# Integration Tests — ON Mode Delegation Edges
# ============================================================


def test_deterministic_pe_vle_async_no_burst_on_mode(
    on_mode, setup_model_and_video, mq_consumer
):
    """ON mode — guard against start-alarm bursts during a sustained incident."""
    PeVle2StageAsyncService = on_mode.PeVle2StageAsyncService
    video_path = setup_model_and_video["video_anomaly"]

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="889", organization="on_mode_burst", categories=["fire_pe_vle_ret"]
    )

    service = PeVle2StageAsyncService(q)
    _run_video_through_service(service, user_param, video_path, fps_div=2)
    stop_thread(q)
    time.sleep(8)

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

    # AnomalyClassifier may reject — 0 starts is allowed. Whatever does fire,
    # it must not burst.
    assert n_starts <= 2, f"ON mode: start-alarm burst ({n_starts} starts) — PE reset regression"
    logger.info(f"ON mode no_burst: {n_starts} starts, {n_ends} ends")


def test_pe_vle_async_torch_one_camera_on_mode(on_mode, setup_model_and_video):
    """ON mode — Stage-1 with torch.Tensor input through the validation_server path."""
    PeVle2StageAsyncService = on_mode.PeVle2StageAsyncService
    video_path = setup_model_and_video["video_anomaly"]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))

    q = Queue(maxsize=1)
    user_param = _make_user_param(
        camera_id="891", organization="on_mode_torch", categories=["fire_pe_vle_ret"]
    )
    sid = _stream_id(user_param)

    PeVle2StageAsyncService(q)
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


# ============================================================
# Debug Tests — Visualization Output
# ============================================================


def _format_status_text(matched_alarm, mode_label: str = "") -> str:
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
    """Stream frames through the queue and overlay PE inference state on each
    frame, writing the result to a debug .mp4."""
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

    assert os.path.exists(debug_path), f"debug video write failed: {debug_path}"
    assert os.path.getsize(debug_path) > 0, f"debug video is empty: {debug_path}"
    return {"starts": n_starts, "ends": n_ends, "total": n_starts + n_ends}


def test_visualize_pe_vle_async_fire_off_mode(off_mode, setup_model_and_video):
    """OFF mode — fire video visualization (Stage-1 publishes directly)."""
    paths = setup_model_and_video
    user_param = _make_user_param(
        camera_id="100", organization="vis_off_fire", categories=["fire_pe_vle_ret"]
    )
    counts = _run_with_visualization(
        off_mode.PeVle2StageAsyncService,
        video_path=paths["video_anomaly"],
        debug_path=paths["debug_anomaly_off"],
        user_param=user_param,
        mode_label="OFF",
    )
    logger.info(
        f"OFF mode fire visualization: {counts['starts']} starts, {counts['ends']} ends "
        f"→ {paths['debug_anomaly_off']}"
    )
    assert counts["starts"] > 0, "fire video produced no start alarms (visualization)"


def test_visualize_pe_vle_async_fire_on_mode(on_mode, setup_model_and_video):
    """ON mode — fire video visualization (validation_server delegation flow)."""
    paths = setup_model_and_video
    user_param = _make_user_param(
        camera_id="900", organization="vis_on_fire", categories=["fire_pe_vle_ret"]
    )
    counts = _run_with_visualization(
        on_mode.PeVle2StageAsyncService,
        video_path=paths["video_anomaly"],
        debug_path=paths["debug_anomaly_on"],
        user_param=user_param,
        mode_label="ON",
    )
    logger.info(
        f"ON mode fire visualization: {counts['starts']} starts, {counts['ends']} ends "
        f"→ {paths['debug_anomaly_on']}"
    )
    assert counts["starts"] > 0, "fire video produced no start alarms (ON visualization)"
    assert counts["starts"] <= 2, f"ON mode visualization: start-alarm burst ({counts['starts']})"
