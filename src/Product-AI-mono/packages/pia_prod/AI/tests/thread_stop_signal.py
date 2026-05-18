import threading
import gc
import time
import torch
from pia_prod.AI.bases.service_base import ServiceBase


def stop_thread(q):
    # inference thread에 종료 신호 전송
    while True:
        if q.empty():
            q.put({"batches": None, "stream_ids": None, "user_params": None})
            break
        time.sleep(1)

    # inference thread가 실제로 종료될 때까지 대기 (GPU/프로세스 리소스 해제 보장)
    for thread in threading.enumerate():
        if thread.name == "object_detection_ai_inference":
            thread.join(timeout=60)

    # Service 인스턴스의 모든 속성을 해제하여 GPU 메모리 반환
    # thread_run_get_logging_state daemon 스레드가 self를 closure로 캡쳐하여 GC가 수거하지 못하므로
    # 인스턴스의 모든 속성을 None으로 설정하여 model, ort_session, trt_engine 등
    # PyTorch/ONNX/TensorRT GPU 리소스를 강제 해제
    # thread_run_get_logging_state daemon 스레드가 접근하는 속성은 보존
    _KEEP_ATTRS = frozenset({"logging_flag", "logging_state_remain_minute"})
    for obj in gc.get_objects():
        try:
            if isinstance(obj, ServiceBase) and obj.analysis_data_queue is q:
                for attr in list(vars(obj).keys()):
                    if attr in _KEEP_ATTRS:
                        continue
                    try:
                        setattr(obj, attr, None)
                    except Exception:
                        pass
        except Exception:
            pass

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return True
