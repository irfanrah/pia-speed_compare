import json
import sys
from queue import Queue
from threading import Thread
from typing import Optional

import pika
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pika.exceptions import AMQPConnectionError, ConnectionClosedByBroker
from retry import retry

from .rabbitMQ_config import RabbitMQConfig


class RabbitMQProducer:
    def __init__(
            self,
            config: RabbitMQConfig,
            logger: Optional[object] = None,
            datalake_manager=None
    ):
        self.config = config
        self.logger = logger
        self.datalake_manager = datalake_manager
        self._connect_to_rabbitmq = retry(
            AMQPConnectionError,
            delay=lambda _: self.config.retry_delay,
            tries=lambda _: self.config.max_retries,
            jitter=(1, 3),
        )(self._connect_to_rabbitmq)
        self.connection = None
        self.channel = None
        self.thread_state = True
        self.result_queue = Queue(maxsize=self.config.alarm_queue_size)
        self.connect_to_rabbitmq()
        self.thread_while_send_message()

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.error(message)
        else:
            print(message)

    def connect_to_rabbitmq(self):
        try:
            self._connect_to_rabbitmq()
        except Exception as e:  # pragma: no cover - safety net
            self._log_error(f"RabbitMQ disconnected: {e}")
            self.close()
            sys.exit(1)

    def _connect_to_rabbitmq(self):
        try:
            self._log_info("Trying to connect to RabbitMQ...")
            credentials = pika.PlainCredentials(
                self.config.user_name, self.config.password
            )
            parameters = pika.ConnectionParameters(
                host=self.config.host,
                port=self.config.port,
                credentials=credentials,
                heartbeat=self.config.heartbeat_interval,
                blocked_connection_timeout=self.config.socket_timeout,
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self._log_info("RabbitMQ producer connected")
            return
        except ConnectionClosedByBroker:
            pass
        except Exception as e:  # pragma: no cover - connection failure
            self._log_error(f"Failed to connect to RabbitMQ: {e}")

    def send_message(self, message: dict, queue_name: Optional[str] = None):
        if not queue_name:
            queue_name = self.config.queue_name
        try:
            if not self.connection or self.connection.is_closed or not self.channel:
                self.connect_to_rabbitmq()
            if not message:
                raise ValueError("Message cannot be empty.")
            message["ts"] = str_UTC_ISO8601_ms_now_time()
            self.channel.basic_publish(
                exchange=self.config.exchange,
                routing_key=queue_name,
                body=json.dumps(message, ensure_ascii=False),
                properties=pika.BasicProperties(delivery_mode=2),
                mandatory=True,
            )
            self._log_info(f"Sent message to {queue_name}: {message}")

        except AMQPConnectionError as e:
            self._log_error(f"AMQP error: {e}")
            self.connect_to_rabbitmq()
            self.send_message(message, queue_name=queue_name)
        except Exception as e:  # pragma: no cover - generic failure
            self._log_error(f"Error sending message: {e}")

    def thread_while_send_message(self):
        def run():
            while self.thread_state:
                message, source, queue_name = self.result_queue.get()
                if all([message is None, source is None, queue_name is None]):
                    continue
                if source == self.config.ai_source_handling_name:
                    message = self.convert_frame_to_AWS_S3(message)
                    self.send_message(
                        message, queue_name=queue_name
                    )
                elif source == self.config.rtsp_source_handling_name:
                    self.send_message(
                        message, queue_name=self.config.rtsp_error_queue_name
                    )

        Thread(target=run, name="rabbitMQ_thread_while_send_message").start()

    def convert_frame_to_AWS_S3(self, message: dict) -> dict:
        s3_save_file_name = ""
        if message["thumbnail"] is not None:
            timestamp = str_UTC_ISO8601_ms_now_time()
            camera_id = message["cameraId"]
            organization = message["organization"]
            s3_save_file_name = f"{camera_id}_{organization}_{timestamp}.jpg"
            if self.datalake_manager :
                self.datalake_manager.upload_image(
                    message["thumbnail"],
                    self.config.s3_thumbnail_bucket_name,
                    s3_save_file_name,
                )
        message["thumbnail"] = s3_save_file_name
        return message

    def close(self):
        if self.connection:
            self.connection.close()
            self._log_info("RabbitMQ connection closed")
