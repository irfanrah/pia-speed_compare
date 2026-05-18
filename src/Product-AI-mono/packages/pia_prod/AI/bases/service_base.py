import time
import uuid
from datetime import datetime, timedelta
from queue import Queue
from collections import defaultdict, deque
from threading import Thread
from abc import ABC, abstractmethod
import torch
from pia.utils.devtools.debug_tools import try_except_only_in_prod_mode, print_only_debug_mode
from typing import final
from pia_prod.AI.DTO.output_handler import match_outputs

from pia.vision.roi.roi_manager import ROIManagerBase
import os
from pia_prod.AI.global_config import (
    ENV_KEY,
    DEV_KEY,
    TEAM_KEY,
    AI_TEAM_KEY,
    USER_PARAM_KEY,
    CV_EVENT_KEY,
    RET_EVENT_KEY,
    VQA_EVENT_KEY,
    LOGGING_STATE_REMAIN_MINUTE,
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    NUM_OF_OBJECT_KEY,
)

from pia_prod.AI.utils.log_templates import (
    logging_send_alarm,
    logging_unrecognized_error,
    logging_logger_status_get_error,
)
from pia_prod.AI.utils.init import redis_manager


class ServiceBase(ABC):
    def __init__(self, analysis_data_queue: Queue):
        self.analysis_data_queue = analysis_data_queue
        self.frame_cnt = 0
        self.logging_flag = False
        self.thread_state = True
        self.is_needed_cvt_color = False
        self.team = os.getenv(TEAM_KEY, None)  # TEAM: ai or es
        self._init_values()

        # Load Model
        self._load_model()
        # load ROI manager
        self.roi_manager = self._load_roi_manager()
        # Event Manager
        self.alarm_event_manager = self._load_event_manager()

        # Inference 쓰레드 생성
        if self.team == AI_TEAM_KEY:  # ai 인퍼런스 서버일 때만 해당 인퍼런스 쓰레드를 생성함
            self.thread_ai_inference()  # Inference
            self.thread_run_get_logging_state()  # get logging state

        self.logging_state_remain_minute = int(
            os.environ.get("LOGGING_STATE_REMAIN_MINUTE", LOGGING_STATE_REMAIN_MINUTE)
        )

        # result Alarm filter
        self.alarm_dict_with_uuid = defaultdict(deque)

    @abstractmethod
    def _init_values(self):
        return

    @abstractmethod
    def _load_model(self):
        return

    @abstractmethod
    def _load_event_manager(self):
        return

    @abstractmethod
    def _detect(self, **keyword):
        """
        Args:
            **keyword: batches, stream_ids, user_params

        Returns (alarms, batches, stream_ids, user_params, is_needed_cvt_color)
        """
        return

    @final
    def _load_roi_manager(self):
        return ROIManagerBase()

    @final
    @try_except_only_in_prod_mode
    def __detect(self, **keyword):
        return self._detect(**keyword)

    @final
    def thread_ai_inference(self) -> None:
        """AI inference를 수행하고 결과를 rabbitMQ에 전송합니다.
        Args:
            batches (List[np.array]): RTSP로부터 받아온 len(batches)개의 이미지
            stream_ids (List[str]): 카메라의 stream_id
            user_params (List[AddStreamModel]): AddStreamModel 데이터클래스에 정의된 파라미터들
        Returns:
            None
        """

        def run():
            while self.thread_state:
                # Step 1. 데이터 수신
                if os.getenv(ENV_KEY, None) == DEV_KEY:
                    self.frame_cnt += 1
                t1 = datetime.now()
                datas = self.analysis_data_queue.get()

                if all(
                    [
                        datas["batches"] is None,
                        datas["stream_ids"] is None,
                        datas["user_params"] is None,
                    ]
                ):
                    break  # 종료 조건 충족 시 인퍼런스 반복문 즉시 종료
                t2 = datetime.now()
                results = self.__detect(**datas)  # Inference
                (
                    self.send_alarm(results) if results else None
                )  # Send Alarm를 Inference 내에서 분리 '25.12.11
                now = datetime.now()
                print_only_debug_mode(
                    f"{now}\tbatches len : {len(datas['batches'])}\t\
frame_cnt : {self.frame_cnt}\t\
stream_ids : {', '.join(s for s in datas['stream_ids'])}\n\
이미지 대기 시간 : {(t2 - t1).total_seconds() * 1000:.0f}ms\t\
이미지 분석 : {(now - t2).total_seconds() * 1000:.0f}ms"
                )

        Thread(target=run, name="object_detection_ai_inference", daemon=True).start()

    def send_alarm(self, results: dict):
        alarms = results[ALARMS_KEY]
        batches = results[BATCHES_KEY]
        stream_ids = results[STREAM_IDS_KEY]
        user_params = results[USER_PARAMS_KEY]
        is_needed_cvt_color = results[IS_NEEDED_CVT_COLOR_KEY]
        num_of_object_list = results.get(NUM_OF_OBJECT_KEY, [])  # peoplecounting

        if len(alarms):
            alarms = self.get_alarm_with_uuid(alarms)
            for idx, (stream_id, (is_start, event_uuid)) in enumerate(alarms.items()):
                if "__" in stream_id:
                    stream_id, category_id = stream_id.split("__")
                else:
                    for event_key in [CV_EVENT_KEY, RET_EVENT_KEY, VQA_EVENT_KEY]:
                        if event_key in user_params[idx][USER_PARAM_KEY]:
                            keys = user_params[idx][USER_PARAM_KEY][event_key].keys()
                            if len(keys):
                                category_id = list(keys)[0]
                                break

                batch_idx = [i for i, v in enumerate(stream_ids) if v == stream_id][0]
                user_param = user_params[batch_idx]  #
                logging_send_alarm(stream_id, is_start, category_id, event_uuid=event_uuid)
                # send message
                thumbnail = batches[batch_idx] if is_start else None
                if (
                    isinstance(thumbnail, torch.Tensor) and not None
                ):  # ES inference 의 경우 입력이 tensor
                    thumbnail = thumbnail.detach().contiguous().cpu().numpy()

                if is_needed_cvt_color and thumbnail is not None:
                    thumbnail = thumbnail[..., ::-1]
                match_outputs(
                    thumbnail,
                    is_start,
                    user_param,
                    self.frame_cnt,
                    event_uuid=event_uuid,
                    category_name=category_id,
                )
        
        if len(num_of_object_list):
            for idx, stream_id in enumerate(stream_ids):
                now_pepole_cnt = num_of_object_list[idx]
                # thumbnail = batches[idx]
                # if isinstance(thumbnail, torch.Tensor):
                #     thumbnail = thumbnail.detach().contiguous().cpu().numpy()
                # if is_needed_cvt_color and thumbnail is not None:
                #     thumbnail = thumbnail[..., ::-1]
                logging_send_alarm(
                    stream_id=stream_id,
                    state=now_pepole_cnt,
                    category_id="peoplecounting",
                )
                

    @staticmethod
    def make_inference_key(camera_id, organization, inference_type: int = "inference"):
        """
        inference_type: 'inference', 'ret', 'vqa'
        """
        return f"{camera_id}-{organization}-{inference_type}"

    @staticmethod
    def is_torch_batches(batches, speed_mode=True):
        if not batches:
            return False

        if speed_mode:
            return isinstance(batches[0], torch.Tensor)

        return all(isinstance(batch, torch.Tensor) for batch in batches)

    def get_alarm_with_uuid(
        self,
        alarms: dict,
    ) -> dict:
        """이 함수는 알람의 중복 True/False를 방지하기 위해 Queue구조를 통해 동일 스트림에 대한 알람을 큐로 저장하여 uuid를 보냅니다.

        Args:
            alarms (dict): {stream_id-organization-inference: True/False, ...} 로 이루어진 알람보내야할 딕셔너리

        Returns:
            dict: 필터링된 알람 딕셔너리
                {
                    stream_id-organization-inference: [True/False, uuid]
                }
        NOTE:
            self.alarm_dict_with_uuid = {
                {
                    stream_id-organization-inference: deque([True/False, uuid])
                }
            }
        """
        result_dict = {}

        if len(alarms) == 0:
            return result_dict

        for stream_id, (alarm_flag, category_id) in alarms.items():
            if alarm_flag:
                now_uuid = str(uuid.uuid4())
                now_info = (True, now_uuid)
                stream_id = f"{stream_id}__{category_id}" if category_id else stream_id
                self.alarm_dict_with_uuid[stream_id].append(now_info)
                result_dict[stream_id] = now_info
            elif alarm_flag is False:
                stream_id = f"{stream_id}__{category_id}" if category_id else stream_id
                if len(self.alarm_dict_with_uuid[stream_id]) == 0:
                    logging_unrecognized_error(
                        "There is no alarm to turn off. can't match True/False"
                    )
                    continue
                _, past_uuid = self.alarm_dict_with_uuid[stream_id].popleft()
                result_dict[stream_id] = (False, past_uuid)
        return result_dict

    @final
    def thread_run_get_logging_state(self):
        def run():
            state = {'active': False, 'start_time': None}

            def reset_logging_state():
                self.logging_flag = False
                state['active'] = False
                state['start_time'] = None

            def should_deactivate():
                if state['start_time']:
                    return (
                        state['active']
                        and state['start_time']
                        and datetime.now() - state['start_time']
                        > timedelta(minutes=self.logging_state_remain_minute)
                    )

            while True:
                try:
                    flag = str(redis_manager.get("logging")).lower() if redis_manager else "false"

                    if flag == "true":
                        if not state['active']:
                            self.logging_flag = True
                            state['active'] = True
                            state['start_time'] = datetime.now()

                    else:
                        reset_logging_state()

                    if should_deactivate():
                        reset_logging_state()
                        redis_manager.set("logging", "false")  # Reset Redis flag to false

                except Exception as e:
                    logging_logger_status_get_error(e)

                time.sleep(1)

        Thread(target=run, name=f"logging_flag_updater_{id(self)}", daemon=True).start()
