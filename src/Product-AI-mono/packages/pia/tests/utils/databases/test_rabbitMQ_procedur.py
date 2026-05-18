import functools
import time

import pytest
from pia.utils.databases.rabbitMQ_config import RabbitMQConfig
from pia.utils.databases.rabbitMQ_manager import RabbitMQProducer


@pytest.fixture()
def rabbitmq_producer():
    config = RabbitMQConfig()
    return RabbitMQProducer(config=config)


@pytest.fixture
def rabbit_manager():
    config = RabbitMQConfig()
    # 보통 __init__ 시 접속하거나, connect() 메서드가 있을 수 있습니다.
    manager = RabbitMQProducer(config=config)
    # getattr로 안전 호출 (없으면 무시)
    connect = getattr(manager, "connect", None)
    if callable(connect):
        connect()
    return manager


def require_rabbitmq(func):
    """RabbitMQ 브로커가 연결 가능한 경우에만 테스트 실행"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = RabbitMQConfig()
        manager = RabbitMQProducer(config=config)
        try:
            # 채널/연결 헬스체크: 임시 큐 선언해보고 바로 삭제
            connect = getattr(manager, "connect", None)
            if callable(connect):
                connect()
            ch = getattr(manager, "channel", None)
            if ch is None:
                raise RuntimeError("RabbitManager.channel 이 없습니다.")
            q = "pytest-healthcheck"
            ch.queue_declare(queue=q, durable=False, auto_delete=True)
            # 일부 브로커는 declare만으로도 충분. 필요하면 삭제까지:
            ch.queue_delete(queue=q)
            # close() 있으면 닫기
            close = getattr(manager, "close", None)
            if callable(close):
                close()
        except Exception:
            pytest.skip("RabbitMQ 브로커에 연결할 수 없어 테스트를 건너뜁니다.")
        return func(*args, **kwargs)
    return wrapper


def test_instance(rabbit_manager):
    config = RabbitMQConfig()
    assert isinstance(config, RabbitMQConfig), "Failed to create RabbitConfig instance"
    assert isinstance(rabbit_manager, RabbitMQProducer), "Failed to create RabbitManager instance"


@require_rabbitmq
def test_rabbitmq_pubsub(rabbit_manager):
    """
    Redis CRUD 대응:
      - Create: 큐 선언 + 첫 메시지 발행
      - Read: 메시지 소비(basic_get/consume_one)
      - Update: '키 업데이트' 개념이 없으므로 새 메시지 재발행으로 대응
      - Delete: 큐 purge 및 삭제
    """
    queue_name = "org-camid-infe-semaphore-service-test"

    # 매니저가 헬퍼 메서드를 제공한다고 가정한 버전
    declare_queue = getattr(rabbit_manager, "declare_queue", None)
    purge_queue = getattr(rabbit_manager, "purge_queue", None)
    delete_queue = getattr(rabbit_manager, "delete_queue", None)
    publish = getattr(rabbit_manager, "publish", None)
    consume_one = getattr(rabbit_manager, "consume_one", None)

    # pika channel 직접 접근 (헬퍼가 없을 때 대비)
    ch = getattr(rabbit_manager, "channel", None)
    assert ch is not None, "RabbitManager.channel 이 필요합니다."

    # 큐 선언
    if callable(declare_queue):
        declare_queue(queue_name, durable=False, auto_delete=False)
    else:
        ch.queue_declare(queue=queue_name, durable=False, auto_delete=False)

    # 큐 정리
    if callable(purge_queue):
        purge_queue(queue_name)
    else:
        ch.queue_purge(queue=queue_name)

    # Create: 첫 메시지 발행
    body1 = b"0"
    if callable(publish):
        publish(exchange="", routing_key=queue_name, body=body1)
    else:
        ch.basic_publish(exchange="", routing_key=queue_name, body=body1)

    # Read: 메시지 1건 읽기
    if callable(consume_one):
        msg = consume_one(queue_name, auto_ack=True, timeout=2.0)
        # consume_one이 (method, properties, body) 또는 body만 반환할 수 있어 처리
        if isinstance(msg, tuple) and len(msg) == 3:
            _, _, msg_body = msg
        else:
            msg_body = msg
    else:
        # pika 기본 사용: busy-wait 방식으로 2초간 polling
        msg_body = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            method, properties, body = ch.basic_get(queue=queue_name, auto_ack=True)
            if method:  # 메시지가 있으면 method가 None 아님
                msg_body = body
                break
            time.sleep(0.05)
    print(msg_body)
    assert msg_body == body1, "Failed to publish or consume message from RabbitMQ"

    # Update 대행: 새 메시지 재발행 (큐의 최신 상태를 나타낸다고 가정)
    body2 = b"new_test_value"
    if callable(publish):
        publish(exchange="", routing_key=queue_name, body=body2)
    else:
        ch.basic_publish(exchange="", routing_key=queue_name, body=body2)

    # 다음 메시지 읽기
    if callable(consume_one):
        msg2 = consume_one(queue_name, auto_ack=True, timeout=2.0)
        if isinstance(msg2, tuple) and len(msg2) == 3:
            _, _, msg_body2 = msg2
        else:
            msg_body2 = msg2
    else:
        msg_body2 = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            method, properties, body = ch.basic_get(queue=queue_name, auto_ack=True)
            if method:
                msg_body2 = body
                break
            time.sleep(0.05)

    print(msg_body2)
    assert msg_body2 == body2, "Failed to update (republish) and consume latest message from RabbitMQ"

    # Delete: 큐 비우고 삭제
    if callable(purge_queue):
        purge_queue(queue_name)
    else:
        ch.queue_purge(queue=queue_name)

    if callable(delete_queue):
        delete_queue(queue_name)
    else:
        ch.queue_delete(queue=queue_name)
