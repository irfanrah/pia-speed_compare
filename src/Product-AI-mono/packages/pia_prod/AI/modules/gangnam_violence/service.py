from pia_prod.AI.bases.service_base import ServiceBase
from queue import Queue
from collections import defaultdict, deque
from pia.ai.tasks.VQA.base import VQAConfig
from pia.ai.model import PiaTorchModel
from pia_prod.AI.modules.gangnam_violence.event import VQAViolenceEventManager
from pia_prod.AI.modules.gangnam_violence.roi_manager import VQAViolenceRoIManager
from pia_prod.AI.modules.gangnam_violence.config import (
    INTERNVL3_MODEL_HF_PATH,
    N_GPUS,
    FRAME_PER_TILE_MAX_NUM,
    DEVICE,
    MODEL_INF_DATA_TYPE,
    VIOLENCE_CATEGORY,
    MAX_NEW_TOKEN,
    NUM_SEGMENTS,
    HOST_IP,
)
from pia_prod.AI.modules.gangnam_violence.prompts import VIOLENCE
from pia_prod.AI.global_config import USER_PARAM_KEY, VQA_EVENT_KEY
import numpy as np
import cv2
import threading


class InternVL3TrtViolenceService(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        self.frame_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.param_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))  # user_param 저장
        self.last_stream_id = None
        self.last_user_param = None
        self.buffer_size = 12
        # 추론용 워커 스레드
        self.inference_thread = None
        self.lock = threading.Lock()
        self.ready_event = threading.Event()  # 12개 쌓였다는 시그널
        # 서비스 로드 후 추론 워커 시작
        # _load model 에서 선언시 작동안함
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

    def _load_model(self):
        self.config = VQAConfig(
            model_path=INTERNVL3_MODEL_HF_PATH,
            device=DEVICE,
            frame_per_tile_max_num=FRAME_PER_TILE_MAX_NUM,
            data_type=MODEL_INF_DATA_TYPE,
            n_gpus=N_GPUS,
            save_cache_dir=INTERNVL3_MODEL_HF_PATH,
            max_new_tokens=MAX_NEW_TOKEN,
            num_segments=NUM_SEGMENTS,
            api_host=HOST_IP,
        )
        self.model = PiaTorchModel(
            target_task="VQA", target_model="internvl3trt", config=self.config
        )

    def _init_values(self):
        pass

    def _load_event_manager(self):
        return VQAViolenceEventManager()

    def _load_roi_manager(self):
        return VQAViolenceRoIManager()

    def _match_category2prompt(self, batches, user_params):
        re_batches = []
        prompts = []
        categories = []
        for frame, user_param in zip(batches, user_params):
            events = user_param[USER_PARAM_KEY][VQA_EVENT_KEY]
            temp_categories = []
            for category in events:
                temp_categories.append(category)
                re_batches.append(frame)
                if category in VIOLENCE_CATEGORY:
                    prompts.append(VIOLENCE)
                else:
                    continue
            categories.append(temp_categories)
        return re_batches, prompts, categories

    def _detect(self, **datas):
        """데이터 수집만 담당 (비동기 - 빠르게 리턴)"""
        self.batches = datas["batches"]
        self.stream_ids = datas["stream_ids"]
        self.user_params = datas["user_params"]
        self.batches = self.roi_manager.process_batches_with_roi(
            self.batches, self.user_params
        )

        with self.lock:
            for batch, stream_id, param in zip(self.batches, self.stream_ids, self.user_params):
                batch_resized = cv2.resize(batch, (448, 448), interpolation=cv2.INTER_CUBIC)
                self.frame_buffers[stream_id].append(batch_resized)
                self.param_buffers[stream_id].append(param)
                if len(self.frame_buffers[stream_id]) == self.buffer_size:
                    self.ready_event.set()

        return None  # 추론은 별도 스레드에서 알람 처리

    def _inference_worker(self):
        """추론 실행 담당 (별도 스레드에서 계속 대기)"""
        while self.running:
            try:
                # 12개 쌓일 때까지 대기
                self.ready_event.wait()
                self.ready_event.clear()

                with self.lock:
                    # 준비된 스트림들 필터링 (12개인 것들)
                    ready_stream_ids = [
                        sid
                        for sid in self.frame_buffers.keys()
                        if len(self.frame_buffers[sid]) == self.buffer_size
                    ]

                    if not ready_stream_ids:
                        continue

                    # 준비된 스트림들의 이미지를 영상으로 변환
                    videos = np.stack(
                        [
                            np.stack(list(self.frame_buffers[sid]), axis=0)  # (12, H, W, C)
                            for sid in ready_stream_ids
                        ]
                    )

                    # user_params 매핑 (가장 마지막 param 사용)
                    ready_user_params = [
                        list(self.param_buffers[sid])[-1] for sid in ready_stream_ids
                    ]

                    # 버퍼 클리어 (lock 안에서)
                    for sid in ready_stream_ids:
                        self.frame_buffers[sid].clear()
                        self.param_buffers[sid].clear()

                # lock 밖에서 추론 실행 (추론 중에도 데이터 수집 가능)
                batches_to_process = [videos[i] for i in range(len(videos))]

                batches_final, prompts, categories = self._match_category2prompt(
                    batches_to_process, ready_user_params
                )

                responses = self.model(batches_final, prompts)
                alarms = self.alarm_event_manager.update(responses, categories, ready_stream_ids)
                # self.send_alarm(alarms, batches_final, ready_stream_ids, ready_user_params)
                self.send_alarm(alarms, self.batches, ready_stream_ids, ready_user_params)

            except Exception as e:
                print(f"Inference worker error: {e}")
                import traceback

                traceback.print_exc()

    def __del__(self):
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
