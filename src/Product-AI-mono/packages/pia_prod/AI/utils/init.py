from pathlib import Path
import sys
import os
from pia.utils.logging.logger import CallLog, resolve_log_basename
from pia.utils.api.datalake import DataLakeManager
from pia.utils.databases import RabbitMQConfig, RabbitMQProducer, RedisConfig, RedisManager
from pia_prod.AI.global_config import (
    ENV_KEY,
    PROD_KEY,
    DEV_KEY,
    ES_TEAM_KEY,
    TEAM_KEY,
    AI_TEAM_KEY,
)


def declare_logger():
    log_path = "logs/"
    log_name = "logs"
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logger = CallLog(log_name=log_name, log_level="debug")
    logger_file_name = resolve_log_basename()  # "File name"
    logger.add_file_handler(path=f"{log_path}{logger_file_name}", H_type="R", backupCount=60)
    logger.add_stream_handler(H_type="stream")
    sys.excepthook = logger.handle_exception
    return logger


def declare_datalake_manager():
    datalake_manager = DataLakeManager(
        access_key_id=os.getenv("BACKEND_S3_ACCESS_KEY"),
        secret_access_key=os.getenv("BACKEND_S3_SECRET_KEY"),
        region_name=os.getenv("BACKEND_S3_THUMBNAIL_BUCKET_REGION"),
        endpoint_url=os.getenv("BACKEND_S3_ENDPOINT"),
    )
    return datalake_manager


def declare_redis_manager():
    redis_manager = RedisManager(
        config=RedisConfig(
            host=os.getenv("REDIS_IP", "localhost"),
            port=os.getenv("REDIS_PORT", "6379"),
            db=os.getenv("REDIS_DB", "0"),
        )
    )
    return redis_manager


def declare_rabbitmq_producer_config():
    config = RabbitMQConfig(
        host=os.getenv("BACKEND_RABBITMQ_IP", "localhost"),
        port=os.getenv("BACKEND_RABBITMQ_PORT", "5672"),
        queue_name="",
        exchange=os.getenv("BACKEND_RABBITMQ_EXCHANGE", ""),
        exchange_type=os.getenv("BACKEND_RABBITMQ_EXCHANGE_TYPE", "direct"),
        user_name=os.getenv("BACKEND_RABBITMQ_USER_NAME", "guest"),
        password=os.getenv("BACKEND_RABBITMQ_PASSWORD", "guest"),
        retry_delay=int(os.getenv("PYTHON_RABBITMQ_RECONNECT_DELAY", 2)),
        max_retries=int(os.getenv("PYTHON_RABBITMQ_MAX_RECONNECT_ATTEMPTS", 5)),
        heartbeat_interval=int(os.getenv("PYTHON_RABBITMQ_HEARBEAT_INTERVAL", 60)),
        socket_timeout=int(os.getenv("PYTHON_RABBITMQ_SOCKET_TIMEOUT", 10)),
        alarm_queue_size=int(os.getenv("PYTHON_RABBITMQ_ALARM_QUEUE_SIZE", 100)),
        rtsp_error_queue_name=os.getenv("BACKEND_RTSP_ERROR_RABBITMQ_QUEUE_NAME", "roi_error/"),
        s3_thumbnail_bucket_name=os.getenv("BACKEND_S3_THUMBNAIL_BUCKET_NAME", "thumbnail"),
    )
    return config


def get_rabbitmq_producer(rabbitmq_config, logger, datalake_manager):
    return RabbitMQProducer(
        config=rabbitmq_config,
        logger=logger,
        datalake_manager=datalake_manager,
    )


now_team = os.getenv(TEAM_KEY, None)  # TEAM: ai or es
now_env = os.getenv(ENV_KEY, DEV_KEY)  # ENV: dev or prod. 검증을 위한 코드가 따로 없으므로 동일하게 적용됨

if now_team == AI_TEAM_KEY:
    logger = declare_logger()
    datalake_manager = declare_datalake_manager()
    redis_manager = declare_redis_manager()
    rabbitmq_config = declare_rabbitmq_producer_config()
    alarm_producer = get_rabbitmq_producer(rabbitmq_config, logger, datalake_manager)

else:  # AI 이외 팀, ES팀 포함
    # TODO : logger 를 제외한 나머지 모듈에 대해 ES팀에서 선언 필요
    logger = declare_logger()
    datalake_manager = None
    redis_manager = None
    alarm_producer = None
