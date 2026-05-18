"""
PE_VQA_2stage Validation Server (자기완결형, 메시징 백엔드 가변).

PE 1차 탐지 결과를 메인 inference로부터 fire-and-forget HTTP POST로 받아
vLLM 기반 VLM으로 2차 검증한 뒤 통과한 알람만 RabbitMQ 또는 Kafka로 발사한다.
사용 모델은 VLLM_MODEL 환경변수로 결정 (default: Qwen/Qwen3.5-0.8B).

설계 원칙
- 외부 패키지(Product-AI-mono의 다른 모듈, KTT 등) 의존성 0
- 자체 pika/confluent-kafka/boto3/redis-py로 메시징/S3/Redis 직접 연동
- 메시지 dict는 다른 PE 기반 모듈과 동일 키 셋 + 백엔드별 형식 자동 변환
- VLM 호출 실패/타임아웃 시 fail-open(통과) — critical 이벤트 보호
- MESSAGING_BACKEND 환경변수로 RabbitMQ(KTT) / Kafka(ES) 선택
"""

import os
import re
import json
import base64
import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import httpx
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prompts import get_validation_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validation_server")

# === 환경 변수 (자체 키 — 외부 사용처가 자체적으로 매핑/주입) ===

# vLLM 연결
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-0.8B")
VLLM_MAX_CONCURRENCY = int(os.getenv("VLLM_MAX_CONCURRENCY", "10"))
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120.0"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "64"))

# 서버 자체
SERVER_HOST = os.getenv("VALIDATION_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("VALIDATION_SERVER_PORT", "8100"))

# 메시징 백엔드 선택 — kafka(ES MACS, default) | rabbitmq(KTT, 명시 override 필요)
# 일차 사용처가 ES MACS이므로 default를 kafka로 둔다. KTT 등 RabbitMQ 환경에서는
# 반드시 MESSAGING_BACKEND=rabbitmq를 명시 설정해야 알람이 정상 발사된다.
MESSAGING_BACKEND = os.getenv("MESSAGING_BACKEND", "kafka").lower()

# RabbitMQ — Product-AI-mono(KTT 등)의 표준 키 사용 (Package-Common-AI-pia_ai_package와 동일)
RABBITMQ_HOST = os.getenv("BACKEND_RABBITMQ_IP", "localhost")
RABBITMQ_PORT = int(os.getenv("BACKEND_RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("BACKEND_RABBITMQ_USER_NAME", "guest")
RABBITMQ_PASS = os.getenv("BACKEND_RABBITMQ_PASSWORD", "guest")
RABBITMQ_EXCHANGE = os.getenv("BACKEND_RABBITMQ_EXCHANGE", "")
RABBITMQ_HEARTBEAT = int(os.getenv("PYTHON_RABBITMQ_HEARBEAT_INTERVAL", "60"))
RABBITMQ_QUEUE_RET = os.getenv("BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME", "ret_queue_dev")

# Kafka — ES MACS 표준 (Avro 스키마 com.macs.events.eventStart, fastavro schemaless 와이어 포맷)
# ES backend는 Schema Registry를 사용하지 않고 fastavro.schemaless_reader로 직접 디코드하므로,
# 여기서도 fastavro.schemaless_writer로 인코드하여 와이어 호환성 보장.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_EVENT_PROCESS = os.getenv("KAFKA_TOPIC_EVENT_PROCESS", "event.process")

# S3 — ES MACS 표준 키
S3_ACCESS_KEY = os.getenv("BACKEND_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("BACKEND_S3_SECRET_KEY", "")
S3_REGION = os.getenv("BACKEND_S3_THUMBNAIL_BUCKET_REGION", "")
S3_ENDPOINT = os.getenv("BACKEND_S3_ENDPOINT", "")
S3_BUCKET = os.getenv("BACKEND_S3_THUMBNAIL_BUCKET_NAME", "thumbnail")

# Redis (UUIDTracker 영속화) — KTT 표준 키
REDIS_HOST = os.getenv("REDIS_IP", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
UUID_TTL_SECONDS = int(os.getenv("UUID_TTL_SECONDS", "3600"))
UUID_KEY_PREFIX = "pe_vqa_2stage:uuid:"

# 정책
PE_VQA_2STAGE_FAIL_OPEN = os.getenv("PE_VQA_2STAGE_FAIL_OPEN", "true").lower() == "true"

# VLM 응답 yes/no 매칭 정규식 — "Yes.", "Yes, fire detected." 같은 부연 응답도 허용
_YES_PATTERN = re.compile(r"\b(yes|1|true)\b", re.IGNORECASE)


# === 요청 모델 (메인 service.py의 _build_validation_payload와 1:1 매칭) ===

class ValidateRequest(BaseModel):
    thumbnail_b64: Optional[str] = None  # base64 encoded JPEG
    is_start: bool
    category_name: str
    stream_id: str
    event_uuid: str
    event_type: str
    user_param: dict[str, Any]  # 메인 측 user_param dict 통째


# === VLM 검증 클라이언트 ===

class VLMValidator:
    def __init__(self):
        self._semaphore = asyncio.Semaphore(VLLM_MAX_CONCURRENCY)
        self._client: Optional[httpx.AsyncClient] = None
        self._api_url = f"{VLLM_API_BASE.rstrip('/')}/chat/completions"

    async def start(self):
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=VLLM_MAX_CONCURRENCY,
                max_keepalive_connections=VLLM_MAX_CONCURRENCY,
            ),
            timeout=VLLM_TIMEOUT,
        )

    async def stop(self):
        if self._client:
            await self._client.aclose()

    async def validate(self, image_b64: str, category_name: str) -> bool:
        prompt = get_validation_prompt(category_name)
        payload = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": VLLM_MAX_TOKENS,
            "temperature": 0.0,
        }
        async with self._semaphore:
            resp = await self._client.post(self._api_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            logger.info(f"VLM 응답: category={category_name}, answer={answer.strip()!r}")
            return bool(_YES_PATTERN.search(answer))


# === Publisher 추상화 (RabbitMQ / Kafka) ===

class MessagePublisher(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def publish(self, message: dict, topic_or_queue: str) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


# === RabbitMQ Publisher (KTT 환경) ===

class RabbitMQPublisher(MessagePublisher):
    def __init__(self):
        import pika  # 백엔드 선택 시에만 import
        self._pika = pika
        self._connection = None
        self._channel = None
        self._lock = threading.Lock()

    def connect(self):
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
        if self._connection is not None and not self._connection.is_closed:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

        credentials = self._pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        params = self._pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=RABBITMQ_HEARTBEAT,
        )
        self._connection = self._pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=RABBITMQ_QUEUE_RET, durable=True)

    def publish(self, message: dict, topic_or_queue: str):
        # Idle 상태에서 RabbitMQ 연결이 heartbeat timeout으로 끊겼는데 is_closed가 False로 남는 경우가 있다.
        # publish 시 StreamLostError/AMQPConnectionError 발생하면 강제 재연결 후 1회 재시도.
        with self._lock:
            body = json.dumps(message, ensure_ascii=False)
            for attempt in range(2):
                try:
                    if attempt > 0 or not self._connection or self._connection.is_closed:
                        self.connect()
                    self._channel.basic_publish(
                        exchange=RABBITMQ_EXCHANGE,
                        routing_key=topic_or_queue,
                        body=body,
                        properties=self._pika.BasicProperties(
                            delivery_mode=2,
                            content_type="application/json",
                        ),
                        mandatory=True,
                    )
                    return
                except (self._pika.exceptions.AMQPConnectionError,
                        self._pika.exceptions.StreamLostError,
                        self._pika.exceptions.ChannelClosed) as e:
                    logger.warning(f"RabbitMQ publish 실패 (재연결 후 재시도): {e}")
                    self._connection = None
                    self._channel = None
                    if attempt == 1:
                        raise

    def close(self):
        with self._lock:
            if self._connection and not self._connection.is_closed:
                self._connection.close()


# === Kafka Publisher (ES MACS 환경) — Avro 직렬화 (fastavro schemaless) ===
#
# ES와 동일한 Avro 스키마 (com.macs.events.eventStart):
# - cameraId: string
# - type: string
# - organization: string
# - name: string
# - isStart: boolean
# - thumbnail: ["null", "string"]  (nullable)
# - incidentThresholdSecond: int
# - incidentTimeoutSecond: int
# - uuid: string
# - ts: long (timestamp-millis, epoch ms)

EVENT_START_AVRO_SCHEMA = {
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


class KafkaPublisher(MessagePublisher):
    """ES MACS 호환 Kafka publisher.

    ES MACS(faststream/aiokafka) 패턴과 동작 정렬:
    - acks=1 (leader-only ack — ES default와 일치)
    - request.timeout.ms=1000 (broker 장애 시 fast-fail)
    - threading.Lock 없음 (confluent-kafka.Producer는 thread-safe)
    - publish마다 flush 안 함 — 100ms 주기 백그라운드 drain task가
      poll(0) + flush(timeout=2)로 처리 (aiokafka 내부 sender task 미러)

    와이어 포맷: fastavro.schemaless_writer (Schema Registry 미사용).
    ES backend의 consumer가 fastavro.schemaless_reader로 직접 디코드하므로 같은 포맷을 사용해야
    상호 호환된다. Confluent Schema Registry 매직 바이트(0x00 + 4-byte schema_id)를 붙이면
    ES consumer가 5바이트만큼 어긋난 바이트로 디코드해 파싱 실패하므로 사용 금지.
    """

    DRAIN_INTERVAL_SECONDS = 0.1
    DRAIN_FLUSH_TIMEOUT = 2.0

    def __init__(self):
        # confluent-kafka는 백엔드가 kafka일 때만 import (RabbitMQ만 쓰는 환경에서 의존성 회피).
        # Producer만 사용하고 직렬화는 fastavro로 직접 수행 (Schema Registry 미사용).
        from confluent_kafka import Producer
        from fastavro import parse_schema, schemaless_writer

        self._Producer = Producer
        self._schemaless_writer = schemaless_writer
        self._parsed_schema = parse_schema(EVENT_START_AVRO_SCHEMA)
        self._producer = None
        self._drain_task: Optional[asyncio.Task] = None
        self._stop_drain = False

    def connect(self):
        self._producer = self._Producer({
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "acks": "1",
            "request.timeout.ms": 1000,
        })

    def _encode(self, message: dict) -> bytes:
        from io import BytesIO

        buf = BytesIO()
        self._schemaless_writer(buf, self._parsed_schema, message)
        return buf.getvalue()

    @staticmethod
    def build_headers(message: dict) -> list[tuple[str, bytes]]:
        # ES MACS `HeaderSerializerMiddleware.to_dict_headers`와 동일하게
        # 빈 값(traceId/traceparent/tracestate/request_uuid)은 부착하지 않는다.
        # 컨슈머가 라우팅에 쓰는 키는 action(필수) + organization(있을 때).
        action = "eventStart" if message.get("isStart") else "eventEnd"
        organization = str(message.get("organization") or "")
        headers: list[tuple[str, bytes]] = [("action", action.encode())]
        if organization:
            headers.append(("organization", organization.encode()))
        return headers

    def publish(self, message: dict, topic_or_queue: str):
        # No lock: confluent-kafka.Producer는 thread-safe.
        # No per-call flush: drain task가 100ms 주기로 처리.
        if self._producer is None:
            self.connect()
        payload = self._encode(message)
        headers = self.build_headers(message)

        def _on_delivery(err, msg):
            if err is not None:
                logger.error(f"Kafka delivery 실패: {err}")
            else:
                logger.debug(
                    f"Kafka delivered: topic={msg.topic()} partition={msg.partition()} "
                    f"offset={msg.offset()}"
                )

        self._producer.produce(
            topic=topic_or_queue,
            value=payload,
            headers=headers,
            on_delivery=_on_delivery,
        )

    async def start_drain(self) -> None:
        """백그라운드 drain task 시작.

        100ms 주기로:
        - poll(0): delivery callback 트리거 (호출 안 하면 callback 큐가 무한 누적)
        - flush(timeout=2): buffered 메시지를 broker로 push

        이 loop이 없으면 callback이 영영 발동 안 하고 producer 내부 buffer가
        cap에 도달할 때까지 자라난다.
        """
        if self._drain_task is not None:
            return
        self._stop_drain = False
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def _drain_loop(self) -> None:
        while not self._stop_drain:
            try:
                await asyncio.sleep(self.DRAIN_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            if self._producer is None:
                continue
            try:
                self._producer.poll(0)
                self._producer.flush(timeout=self.DRAIN_FLUSH_TIMEOUT)
            except Exception as e:
                logger.warning(f"Kafka drain iteration 실패: {e}")

    async def stop_drain(self) -> None:
        """lifespan 종료 시 graceful drain 중지."""
        self._stop_drain = True
        if self._drain_task is not None:
            try:
                await asyncio.wait_for(self._drain_task, timeout=2)
            except asyncio.TimeoutError:
                self._drain_task.cancel()
            except Exception:
                pass
            self._drain_task = None

    def close(self):
        # 종료 시 final flush — drain task는 이미 stop된 상태여야 함.
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
            except Exception:
                pass


# === S3 Uploader ===

class S3Uploader:
    def __init__(self):
        self._client = None
        # 빈 credential short-circuit — boto3 client 자체를 만들지 않음.
        # 만들어 두면 매 업로드가 AWS 기본 endpoint(s3.amazonaws.com)로
        # fallback해 DNS + TLS + SigV4 + retry 후 ~2s 만에 실패하고,
        # 그 시간이 /api/v1/validate 응답에 그대로 누적됨.
        # 본 핸들러는 S3 업로드를 best-effort로 다루며(메시지에 키만
        # 박고 진행), 다운스트림 컨슈머는 placeholder 썸네일로 fallback
        # 하므로 skip은 안전.
        self._enabled = bool(S3_ACCESS_KEY and S3_SECRET_KEY)
        self._disabled_logged = False

    def connect(self):
        session = boto3.session.Session(
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
        )
        self._client = session.client("s3", endpoint_url=S3_ENDPOINT or None)

    @staticmethod
    def build_key(camera_id: str, organization: str, ts_epoch_ms: int) -> str:
        # ES MACS `MQProducer._run_drain_loop`와 동일한 키 형식.
        # date_prefix는 메시지 ts가 아닌 발사 시점 UTC now 기준(ES와 동일).
        date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{date_prefix}/{camera_id}_{organization}_{ts_epoch_ms}.jpg"

    def upload_jpeg(
        self, jpeg_bytes: bytes, camera_id: str, organization: str, ts_epoch_ms: int
    ) -> str:
        filename = self.build_key(camera_id, organization, ts_epoch_ms)
        if not self._enabled:
            # disabled 상태가 silent하지 않도록 첫 skip에 한 번만 경고.
            # 운영자가 S3 업로드를 기대했다면 이 로그로 disabled 상태 확인 가능.
            if not self._disabled_logged:
                logging.getLogger("pe_vqa_2stage_validation").warning(
                    "[S3] disabled (empty BACKEND_S3_ACCESS_KEY/SECRET_KEY) — "
                    "uploads will be skipped, key=%s reserved on the message; "
                    "set credentials to enable real uploads.",
                    filename,
                )
                self._disabled_logged = True
            return filename
        if not self._client:
            self.connect()
        self._client.put_object(
            Body=jpeg_bytes,
            Bucket=S3_BUCKET,
            Key=filename,
            ContentType="image/jpeg",
        )
        return filename


# === UUIDTracker (Redis 기반 — 컨테이너 재시작 시 휘발 방지) ===

class RedisUUIDTracker:
    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._client: Optional[redis.Redis] = None

    def connect(self):
        self._client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
        self._client.ping()

    def _key(self, uuid: str) -> str:
        return f"{UUID_KEY_PREFIX}{uuid}"

    def add(self, uuid: str):
        if self._client is None:
            self.connect()
        self._client.set(self._key(uuid), "1", ex=self._ttl)

    def contains(self, uuid: str) -> bool:
        if self._client is None:
            self.connect()
        return bool(self._client.exists(self._key(uuid)))

    def remove(self, uuid: str):
        if self._client is None:
            self.connect()
        self._client.delete(self._key(uuid))


# === 메시지 빌더 (다른 PE 모듈과 동일 키 셋 + ts 자동 포함) ===

def _utc_iso8601_ms_now() -> str:
    """KTT pia.utils.api.timestamp.str_UTC_ISO8601_ms_now_time과 동일 형식.

    형식: 'YYYY-MM-DDTHH:MM:SS.ssssss' (microsecond 6자리, Z suffix 없음)
    예: '2025-08-27T15:49:00.123456'
    """
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds")


def make_alarm_message_compatible(
    user_param: dict,
    thumbnail_filename: str,
    is_start: bool,
    category_name: str,
    event_type: str,
    event_uuid: str,
    ts_epoch_ms: Optional[int] = None,
) -> dict:
    """백엔드별 알람 메시지 dict 생성.

    RabbitMQ (KTT): 다른 PE 기반 모듈과 동일 — cameraId int, ts ISO8601 microsecond
    Kafka (ES MACS): Avro 스키마 — cameraId string, ts epoch ms long, thumbnail nullable

    ts_epoch_ms를 외부에서 주입하면 message["ts"]와 S3 키의 timestamp가 동일해진다
    (ES MQProducer가 message.ts를 그대로 S3 키에 사용하는 동작과 일치).
    """
    up = user_param["user_param"]
    cat_cfg = up[event_type][category_name]

    base = {
        "type": event_type,
        "organization": up["organization"],
        "name": cat_cfg["name"],
        "isStart": is_start,
        "incidentThresholdSecond": cat_cfg["incidentThresholdSecond"],
        "incidentTimeoutSecond": cat_cfg["incidentTimeoutSecond"],
        "uuid": event_uuid,
    }

    if MESSAGING_BACKEND == "kafka":
        # ES MACS Avro 호환 — cameraId string, ts epoch ms long, thumbnail string("" 허용).
        # ES inference도 빈 thumbnail을 ""(string union 브랜치)로 보내므로 동일 브랜치 사용.
        ts = (
            ts_epoch_ms
            if ts_epoch_ms is not None
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )
        return {
            **base,
            "cameraId": str(up["cameraId"]),                                        # string
            "thumbnail": thumbnail_filename if thumbnail_filename else "",          # string, "" 허용
            "ts": ts,                                                               # long, epoch ms
        }
    else:
        # KTT 표준 (다른 PE 모듈과 동일)
        return {
            **base,
            "cameraId": up["cameraId"],                                              # int 보존
            "thumbnail": thumbnail_filename,                                          # 빈 문자열 허용
            "ts": _utc_iso8601_ms_now(),                                             # ISO8601 microsecond
        }


def resolve_topic_or_queue() -> str:
    """백엔드별 발사 대상 (RabbitMQ queue 또는 Kafka topic)."""
    if MESSAGING_BACKEND == "kafka":
        return KAFKA_TOPIC_EVENT_PROCESS
    return RABBITMQ_QUEUE_RET


def _build_publisher() -> MessagePublisher:
    if MESSAGING_BACKEND == "kafka":
        return KafkaPublisher()
    elif MESSAGING_BACKEND == "rabbitmq":
        return RabbitMQPublisher()
    else:
        raise ValueError(
            f"MESSAGING_BACKEND 값이 잘못됨: {MESSAGING_BACKEND!r} "
            f"(rabbitmq | kafka 중 하나여야 함)"
        )


# === FastAPI App ===

vlm_validator = VLMValidator()
mq_publisher: MessagePublisher = _build_publisher()
s3_uploader = S3Uploader()
uuid_tracker = RedisUUIDTracker(ttl_seconds=UUID_TTL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Validation server backend: {MESSAGING_BACKEND}")
    await vlm_validator.start()
    backend_label = "Kafka" if MESSAGING_BACKEND == "kafka" else "RabbitMQ"
    for component_name, connect_fn in (
        (backend_label, mq_publisher.connect),
        ("S3", s3_uploader.connect),
        ("Redis", uuid_tracker.connect),
    ):
        try:
            connect_fn()
        except Exception as e:
            logger.warning(f"{component_name} 연결 실패 (서버는 계속 동작, 발사 시점에 재시도): {e}")

    # Kafka 백엔드는 백그라운드 drain task 시작 (poll/flush 100ms 주기).
    # RabbitMQ 백엔드는 hasattr 체크로 자동 no-op.
    if hasattr(mq_publisher, "start_drain"):
        await mq_publisher.start_drain()

    logger.info("Validation server started")
    yield
    await vlm_validator.stop()
    if hasattr(mq_publisher, "stop_drain"):
        await mq_publisher.stop_drain()
    mq_publisher.close()
    logger.info("Validation server stopped")


app = FastAPI(title="PE VQA 2-Stage Validation Server", lifespan=lifespan)


@app.post("/api/v1/validate")
async def validate_alarm(req: ValidateRequest):
    """
    PE 1차 알람을 받아 VLM 2차 검증 후 RabbitMQ로 발사.

    is_start=True:
      - VLM 검증 → "yes"면 Redis UUID 등록 + S3 업로드 + RabbitMQ publish
      - VLM 호출 예외 시 PE_VQA_2STAGE_FAIL_OPEN=true(기본)면 통과 처리
    is_start=False:
      - Redis UUID 매칭 시 RabbitMQ publish + UUID 삭제
    """
    # message["ts"]와 S3 키 timestamp를 동일하게 맞추기 위해 한 번만 생성.
    # ES MQProducer는 message.ts를 그대로 S3 키에 사용하므로 동일 동작 보장.
    ts_epoch_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if req.is_start:
        if req.thumbnail_b64 is None:
            raise HTTPException(status_code=400, detail="is_start=True일 때 thumbnail_b64 필수")

        try:
            is_real = await vlm_validator.validate(req.thumbnail_b64, req.category_name)
        except Exception as e:
            if PE_VQA_2STAGE_FAIL_OPEN:
                logger.warning(
                    f"VLM 호출 실패 — fail-open으로 통과 처리: stream={req.stream_id}, "
                    f"uuid={req.event_uuid}, error={e}"
                )
                is_real = True
            else:
                logger.error(f"VLM 검증 실패 (폐기): stream={req.stream_id}, error={e}")
                return {"status": "discarded", "uuid": req.event_uuid, "reason": "vlm_error"}

        if not is_real:
            logger.info(
                f"False positive 폐기: stream={req.stream_id}, "
                f"category={req.category_name}, uuid={req.event_uuid}"
            )
            return {"status": "discarded", "uuid": req.event_uuid, "reason": "vlm_rejected"}

        try:
            uuid_tracker.add(req.event_uuid)
        except Exception as e:
            logger.error(f"Redis UUID 등록 실패: {e}")

        # ES 정책과 동일 — 낙관적: 키는 항상 박고 업로드는 best-effort.
        # 실패해도 메시지엔 키가 들어가고, 컨슈머는 깨진 썸네일 fallback 책임.
        up = req.user_param["user_param"]
        thumbnail_filename = S3Uploader.build_key(
            str(up["cameraId"]), up["organization"], ts_epoch_ms
        )
        try:
            jpeg_bytes = base64.b64decode(req.thumbnail_b64)
            await asyncio.get_running_loop().run_in_executor(
                None,
                s3_uploader.upload_jpeg,
                jpeg_bytes,
                str(up["cameraId"]),
                up["organization"],
                ts_epoch_ms,
            )
        except Exception as e:
            logger.error(f"S3 업로드 실패 (메시지엔 키만 박고 진행): {e}")

    else:
        try:
            registered = uuid_tracker.contains(req.event_uuid)
        except Exception as e:
            logger.error(f"Redis UUID 조회 실패: {e}")
            registered = False
        if not registered:
            logger.info(
                f"종료 알람 폐기 (시작 미검증): stream={req.stream_id}, uuid={req.event_uuid}"
            )
            return {
                "status": "discarded",
                "uuid": req.event_uuid,
                "reason": "start_not_validated",
            }
        try:
            uuid_tracker.remove(req.event_uuid)
        except Exception as e:
            logger.error(f"Redis UUID 삭제 실패: {e}")
        thumbnail_filename = ""

    try:
        message = make_alarm_message_compatible(
            user_param=req.user_param,
            thumbnail_filename=thumbnail_filename,
            is_start=req.is_start,
            category_name=req.category_name,
            event_type=req.event_type,
            event_uuid=req.event_uuid,
            ts_epoch_ms=ts_epoch_ms,
        )
    except KeyError as e:
        logger.error(f"user_param 키 누락: {e}, payload={req.user_param}")
        return {"status": "error", "uuid": req.event_uuid, "reason": f"missing_key:{e}"}

    target = resolve_topic_or_queue()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, mq_publisher.publish, message, target)
    except Exception as e:
        logger.error(f"{MESSAGING_BACKEND} 발사 실패: {e}")
        return {"status": "error", "uuid": req.event_uuid, "reason": str(e)}

    logger.info(
        f"알람 발사 완료: stream={req.stream_id}, category={req.category_name}, "
        f"uuid={req.event_uuid}, is_start={req.is_start}"
    )
    return {"status": "sent", "uuid": req.event_uuid}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
