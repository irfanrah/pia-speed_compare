# Redis 서버에 연결
import json

import redis
from pia.utils.databases.redis_config import RedisConfig
from redis.backoff import ExponentialBackoff
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError
from redis.retry import Retry


class RedisManager:
    retry_strategy = Retry(
        backoff=ExponentialBackoff(),
        retries=5,
    )
    def __init__(self, config:RedisConfig):
        self.config = config
        self.client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            retry=RedisManager.retry_strategy,
            retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
            socket_keepalive=True,
            health_check_interval=60,
        )

    def set(self, key, value, expire: int = None):
        self.client.set(key, json.dumps(value, ensure_ascii=False), ex=expire)

    def get(self, key):
        try:
            return json.loads(self.client.get(key))
        except TypeError:
            # Can't find key in redis
            return None

    def delete(self, key):
        self.client.delete(key)

    def update(self, key, value: dict, expire: int = None):
        data = self.get(key) or {}
        for dict_key, dict_value in value.items():
            data[dict_key] = dict_value
        self.set(key, data, expire=expire)

    def get_all(self, pattern=None):
        return self.client.keys(pattern)

    def decrease_int(self, key, expire_second, amount=1):
        self.client.decr(key, amount=amount)
        self.client.expire(key, time=expire_second)
