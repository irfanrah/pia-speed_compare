import functools

import pytest
from pia.utils.databases.redis_config import RedisConfig
from pia.utils.databases.redis_manager import RedisManager


@pytest.fixture
def redis_manager():
    config = RedisConfig()
    return RedisManager(config=config)


def require_redis_ping(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = RedisConfig()
        redis_manager = RedisManager(config=config)
        try:
            redis_manager.client.ping()
        except Exception:
            pytest.skip("Redis 서버에 연결할 수 없어 테스트를 건너뜁니다.")
        return func(*args, **kwargs)
    return wrapper


def test_instance(redis_manager):
    config = RedisConfig()
    assert isinstance(config, RedisConfig), "Failed to create RedisConfig instance"
    assert isinstance(redis_manager, RedisManager), "Failed to create RedisManager instance"


@require_redis_ping
def test_redis_crud(redis_manager):
    test_key = "org-camid-infe-semaphore-service"
    test_value = 0

    # Test Create
    redis_manager.set(key=test_key, value=test_value, expire=10)
    assert redis_manager.get(test_key) == test_value, "Failed to set or get value in Redis"

    # Test Update
    new_value = "new_test_value"
    redis_manager.set(test_key, new_value)
    assert redis_manager.get(test_key) == new_value, "Failed to update value in Redis"

    # Test Delete
    redis_manager.delete(test_key)
    assert redis_manager.get(test_key) is None, "Failed to delete key in Redis"
