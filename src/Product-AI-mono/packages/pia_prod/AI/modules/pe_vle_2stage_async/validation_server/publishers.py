"""
Self-contained publish stack for the PE-VLE validation server.

Components
----------
  MessagePublisher  ABC for the two backends
  RabbitMQPublisher pika BlockingConnection + reconnect-on-publish
  KafkaPublisher    confluent_kafka + fastavro schemaless wire format
  S3Uploader        thumbnail JPEG → S3 (best-effort; key always reserved)
  RedisUUIDTracker  is_start/is_end pairing dedup with TTL

Plus message-builder helpers:
  utc_iso8601_ms_now           → KTT timestamp format
  make_alarm_message_compatible → backend-aware alarm dict
  resolve_topic_or_queue       → pick the right destination per backend
  build_publisher              → factory selecting RabbitMQ vs Kafka
"""

import asyncio
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

import boto3
import redis

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_EVENT_PROCESS,
    MESSAGING_BACKEND,
    RABBITMQ_EXCHANGE,
    RABBITMQ_HEARTBEAT,
    RABBITMQ_HOST,
    RABBITMQ_PASS,
    RABBITMQ_PORT,
    RABBITMQ_QUEUE_RET,
    RABBITMQ_USER,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_REGION,
    S3_SECRET_KEY,
    UUID_KEY_PREFIX,
)

logger = logging.getLogger("pe_vle_validation_server.publishers")


# === Avro schema for Kafka (matches ES MACS com.macs.events.eventStart) =====
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


# === Publisher ABC ==========================================================
class MessagePublisher(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def publish(self, message: dict, topic_or_queue: str) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


# === RabbitMQ Publisher (KTT) ==============================================
class RabbitMQPublisher(MessagePublisher):
    """pika BlockingConnection wrapped with a re-connect-on-publish retry.

    Idle RabbitMQ connections sometimes drop on heartbeat timeout while
    `is_closed` still reports False; the next publish surfaces this as
    AMQPConnectionError / StreamLostError / ChannelClosed. We catch those,
    rebuild the connection, and retry once."""

    def __init__(self):
        import pika  # only import when this backend is selected
        self._pika = pika
        self._connection = None
        self._channel = None
        self._lock = threading.Lock()

    def connect(self) -> None:
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

    def publish(self, message: dict, topic_or_queue: str) -> None:
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
                except (
                    self._pika.exceptions.AMQPConnectionError,
                    self._pika.exceptions.StreamLostError,
                    self._pika.exceptions.ChannelClosed,
                ) as e:
                    logger.warning(f"RabbitMQ publish failed (retrying after reconnect): {e}")
                    self._connection = None
                    self._channel = None
                    if attempt == 1:
                        raise

    def close(self) -> None:
        with self._lock:
            if self._connection and not self._connection.is_closed:
                self._connection.close()


# === Kafka Publisher (ES MACS) =============================================
class KafkaPublisher(MessagePublisher):
    """confluent-kafka producer with fastavro schemaless wire format.

    Aligns with ES MACS pattern (faststream/aiokafka equivalent behavior):
    - acks=1 (leader-only ack — matches ES default)
    - request.timeout.ms=1000 (fast fail under broker degradation)
    - No threading.Lock — confluent-kafka.Producer is thread-safe
    - No per-call flush — a background drain task polls()/flushes() every 100ms
      (mirrors aiokafka's internal sender task pattern)

    No Schema Registry. ES backend's consumer reads with
    `fastavro.schemaless_reader`, so the producer must match. Confluent's
    magic-byte wire format would shift the consumer five bytes off and
    break decode."""

    DRAIN_INTERVAL_SECONDS = 0.1
    DRAIN_FLUSH_TIMEOUT = 2.0

    def __init__(self):
        from confluent_kafka import Producer
        from fastavro import parse_schema, schemaless_writer

        self._Producer = Producer
        self._schemaless_writer = schemaless_writer
        self._parsed_schema = parse_schema(EVENT_START_AVRO_SCHEMA)
        self._producer = None
        self._drain_task: Optional[asyncio.Task] = None
        self._stop_drain = False

    def connect(self) -> None:
        self._producer = self._Producer({
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "acks": "1",
            "request.timeout.ms": 1000,
        })

    def _encode(self, message: dict) -> bytes:
        buf = BytesIO()
        self._schemaless_writer(buf, self._parsed_schema, message)
        return buf.getvalue()

    @staticmethod
    def build_headers(message: dict) -> list:
        # ES MACS HeaderSerializerMiddleware contract: omit empty values;
        # only `action` is mandatory, `organization` is added when present.
        action = "eventStart" if message.get("isStart") else "eventEnd"
        organization = str(message.get("organization") or "")
        headers = [("action", action.encode())]
        if organization:
            headers.append(("organization", organization.encode()))
        return headers

    def publish(self, message: dict, topic_or_queue: str) -> None:
        # No lock: confluent-kafka.Producer is thread-safe.
        # No per-call flush: drain task handles it every 100ms.
        if self._producer is None:
            self.connect()
        payload = self._encode(message)
        headers = self.build_headers(message)

        def _on_delivery(err, msg):
            if err is not None:
                logger.error(f"Kafka delivery failed: {err}")
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
        """Start the background drain task.

        Periodically (every 100ms):
        - poll(0): triggers delivery callbacks (otherwise they queue forever)
        - flush(timeout=2): pushes buffered messages to broker

        Without this loop, callbacks never fire and the producer's internal
        buffer keeps growing until it caps out.
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
                logger.warning(f"Kafka drain iteration failed: {e}")

    async def stop_drain(self) -> None:
        """Stop the drain task gracefully (called from lifespan teardown)."""
        self._stop_drain = True
        if self._drain_task is not None:
            try:
                await asyncio.wait_for(self._drain_task, timeout=2)
            except asyncio.TimeoutError:
                self._drain_task.cancel()
            except Exception:
                pass
            self._drain_task = None

    def close(self) -> None:
        # Final synchronous flush on shutdown — drain task should be stopped first.
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
            except Exception:
                pass


# === S3 Uploader ============================================================
class S3Uploader:
    def __init__(self):
        self._client = None
        # Empty credentials short-circuit: don't build a boto3 client at all.
        # Otherwise every upload would fall through to the AWS default endpoint
        # (s3.amazonaws.com) and pay ~2s of DNS + TLS + SigV4 + retry per
        # request before failing. The validator's wire contract treats S3 upload
        # as best-effort (the message reserves the key regardless of upload
        # success), so skipping is safe — downstream consumers fall back to a
        # placeholder thumbnail.
        self._enabled = bool(S3_ACCESS_KEY and S3_SECRET_KEY)
        self._disabled_logged = False

    def connect(self) -> None:
        session = boto3.session.Session(
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
        )
        self._client = session.client("s3", endpoint_url=S3_ENDPOINT or None)

    @staticmethod
    def build_key(camera_id: str, organization: str, ts_epoch_ms: int) -> str:
        # Same key shape ES MQProducer uses: <YYYYMMDD>/<cam>_<org>_<ts>.jpg.
        # Date prefix is the UTC now of the upload, not the message ts.
        date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{date_prefix}/{camera_id}_{organization}_{ts_epoch_ms}.jpg"

    def upload_jpeg(
        self, jpeg_bytes: bytes, camera_id: str, organization: str, ts_epoch_ms: int
    ) -> str:
        filename = self.build_key(camera_id, organization, ts_epoch_ms)
        if not self._enabled:
            # Log-once so the disabled state is visible in operations without
            # adding per-request noise. Operators who expected uploads to land
            # in S3 will see this line in startup logs (or first traffic).
            if not self._disabled_logged:
                logging.getLogger("pe_vle_validation_server").warning(
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


# === Redis UUID tracker =====================================================
class RedisUUIDTracker:
    """Tracks `event_uuid`s that passed Stage-2 validation. End-state alarms
    only publish if their start was registered here, mirroring pe_vqa's
    pairing dedup. Survives container restarts (TTL'd keys in Redis)."""

    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._client: Optional[redis.Redis] = None

    def connect(self) -> None:
        self._client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
        self._client.ping()

    def _key(self, uuid: str) -> str:
        return f"{UUID_KEY_PREFIX}{uuid}"

    def add(self, uuid: str) -> None:
        if self._client is None:
            self.connect()
        self._client.set(self._key(uuid), "1", ex=self._ttl)

    def contains(self, uuid: str) -> bool:
        if self._client is None:
            self.connect()
        return bool(self._client.exists(self._key(uuid)))

    def remove(self, uuid: str) -> None:
        if self._client is None:
            self.connect()
        self._client.delete(self._key(uuid))


# === Message builders =======================================================
def utc_iso8601_ms_now() -> str:
    """KTT-shaped timestamp: 'YYYY-MM-DDTHH:MM:SS.ssssss' (no Z suffix)."""
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
    """Build the alarm dict the consumer expects.

    RabbitMQ (KTT): same shape as the in-process pia_prod producer —
        cameraId int, ts ISO8601 microsecond.
    Kafka (ES MACS): Avro-shaped — cameraId string, ts epoch ms long,
        thumbnail required string ("" when empty).

    Pass `ts_epoch_ms` from the caller so message["ts"] and the S3 key
    timestamp align (matches ES MQProducer's behavior of reusing message.ts
    for the S3 key).
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
        ts = (
            ts_epoch_ms
            if ts_epoch_ms is not None
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )
        return {
            **base,
            "cameraId": str(up["cameraId"]),
            "thumbnail": thumbnail_filename if thumbnail_filename else "",
            "ts": ts,
        }
    return {
        **base,
        "cameraId": up["cameraId"],
        "thumbnail": thumbnail_filename,
        "ts": utc_iso8601_ms_now(),
    }


def resolve_topic_or_queue() -> str:
    """Backend-specific publish target."""
    if MESSAGING_BACKEND == "kafka":
        return KAFKA_TOPIC_EVENT_PROCESS
    return RABBITMQ_QUEUE_RET


def build_publisher() -> MessagePublisher:
    if MESSAGING_BACKEND == "kafka":
        return KafkaPublisher()
    if MESSAGING_BACKEND == "rabbitmq":
        return RabbitMQPublisher()
    raise ValueError(
        f"MESSAGING_BACKEND invalid: {MESSAGING_BACKEND!r} "
        f"(must be 'rabbitmq' or 'kafka')"
    )
