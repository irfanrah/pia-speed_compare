from pia.utils.api.hugging_face import HFModelDownloader
from pia.utils.api.nas import NasManager, NasConfig
from pia_prod.AI.utils.init import redis_manager, alarm_producer
import pytest
import threading
import os
import redis


ASSETS_SAVE_DIR = "assets"
ASSETS_MODEL_SAVE_DIR = os.path.join(ASSETS_SAVE_DIR, "model")
ASSETS_VIDEO_SAVE_DIR = os.path.join(ASSETS_SAVE_DIR, "videos")


@pytest.fixture(scope="session")
def hf_downloader():
    return HFModelDownloader(
        namespace=os.getenv("HF_NAMESPACE", "PIA-SPACE-LAB"),
        auth_token=os.getenv("HF_AUTH_TOKEN"),
        cache_dir=ASSETS_MODEL_SAVE_DIR,
    )


@pytest.fixture(scope="session")
def nas_downloader():
    return NasManager(NasConfig(NAS_PATH_FILE="http://172.168.47.36/Product_ai_mono/test_videos"))


@pytest.fixture(scope="session")
def video_save_dir():
    return ASSETS_VIDEO_SAVE_DIR


@pytest.fixture(autouse=True)
def set_env():
    os.environ["TEAM"] = "ai"


@pytest.fixture(scope="session", autouse=True)
def cleanup_alarm_producer():
    """테스트 세션 종료 후 alarm_producer non-daemon 스레드를 정리하여 pytest가 정상 종료되도록 함.
    alarm_producer는 세션 전체에서 공유되는 싱글턴이므로 각 테스트가 아닌 세션 끝에서 한 번만 종료.
    """
    yield
    if alarm_producer is not None:
        alarm_producer.thread_state = False
        alarm_producer.result_queue.put([None, None, None])
        for thread in threading.enumerate():
            if thread.name == "rabbitMQ_thread_while_send_message":
                thread.join(timeout=30)


def set_logging_flag(flag: bool = True) -> bool:
    try:
        if flag:
            redis_manager.set("logging", "true")
        else:
            redis_manager.set("logging", "false")
    except redis.RedisError:
        print("Failed to set logging flag in Redis")
        return False
    return True
