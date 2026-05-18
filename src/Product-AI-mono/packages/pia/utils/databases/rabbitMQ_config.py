from dataclasses import dataclass


@dataclass
class RabbitMQConfig:
    """Configuration values for :class:`RabbitMQProducer`.

    All values have sensible defaults so that unit tests or local
    development can run without external dependencies. Production
    deployments are expected to provide the appropriate values
    through this config object.
    """

    host: str = "localhost"
    port: int = 5672
    queue_name: str = ""
    user_name: str = "guest"
    password: str = "guest"
    exchange: str = ""
    exchange_type: str = "direct"
    retry_delay: int = 5
    max_retries: int = 5
    heartbeat_interval: int = 60
    socket_timeout: int = 5
    alarm_queue_size: int = 10
    ai_source_handling_name: str = "ai"
    rtsp_source_handling_name: str = "rtsp"
    rtsp_error_queue_name: str = ""
    s3_thumbnail_bucket_name: str = ""
