from pia_prod.AI.utils.init import logger


############################## RabbitMQ ############################## noqa
def logging_rabbitmq_initialization():
    """
    RabbitMQ 초기화 로그 기록.
    """
    logger.info("RabbitMQ initialization")


def logging_rabbitmq_try_to_connect():
    """
    RabbitMQ 연결 시도 로그 기록.
    """
    logger.info("Trying to connect to RabbitMQ...")


def logging_rabbitmq_producer_connected():
    """
    RabbitMQ 연결 성공 로그 기록.
    """
    logger.info("Successfully connected to RabbitMQ.")


def logging_rabbitmq_producer_filed_to_connect(e):
    """
    RabbitMQ 연결 실패 로그 기록.

    Args:
        e (Exception): 발생한 예외.
    """
    logger.warning(f"Failed to connect to RabbitMQ retring... {e}")


def logging_rabbitmq_producer_disconnected(e):
    """
    최대 재연결 시도 후 RabbitMQ 연결이 실패했음을 나타내는 로그 기록.

    Args:
        e (Exception): 발생한 예외.
    """
    logger.info(f"Failed to connect to RabbitMQ after maximum retries. : {e}")


def logging_rabbitmq_producer_send_message(message, queue_name):
    logger.info(f"Sent {queue_name} Queue : {message} ")


def logging_rabbitmq_producer_send_message_error(message):
    """
    RabbitMQ 메시지 전송 실패 로그 기록.

    Args:
        message (dict | str): 전송 실패한 메시지.
    """
    logger.error(f"Failed to send message: {message}")


def logging_rabbitmq_producer_failed_due_to_send_message(e):
    """
    RabbitMQ 메시지 전송 중 연결 오류가 발생했을 때 로그 기록.

    Args:
        e (Exception): 발생한 예외.
    """
    logger.info(f"Failed to send message due to connection error: {e}. Retrying...")


def logging_rabbitmq_producer_send_message_retry_complete():
    """
    RabbitMQ 메시지 재전송 성공 로그 기록.
    """
    logger.info("rabbitMQ producer send message retry complete")


def logging_rabbitmq_connection_closed():
    """
    RabbitMQ 연결이 종료되었음을 나타내는 로그 기록.
    """
    logger.info("rabbitMQ connection closed")


###################################################################### noqa


def logging_send_alarm(stream_id, state, category_id=None, event_uuid=None):
    if category_id is not None:
        logger.info(
            f"Send Alarm  : stream_id: {stream_id} state: {state} "
            + f"category_id: {category_id} uuid: {event_uuid}"
        )
    else:
        logger.info(f"Send Alarm  : stream_id: {stream_id} state: {state}, uuid: {event_uuid}")


def logging_unrecognized_error(e):
    logger.error(f"{e}")


def logging_logger_status_get_error(e):
    logger.error(f"Failed to get logging status from redis: {e}")
