from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.DaeGu_crowd_people.event import CrowdPeopleEvent
from queue import Queue
from pia.ai.tasks.CP.base import CPONNXConfig
from pia.ai.model import PiaTorchModel
from pia_prod.AI.modules.DaeGu_crowd_people.config import CROWD_PEOPLE_ONNX_MODEL_PATH, DEVICE
from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class CPService(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        self.is_needed_cvt_color = True

    def _load_model(self):
        config = CPONNXConfig(model_path=CROWD_PEOPLE_ONNX_MODEL_PATH, device=DEVICE)
        self.model = PiaTorchModel(target_task="CP", target_model="clip_ebc_onnx", config=config)

    def _init_values(self):
        pass

    def _load_event_manager(self):
        return CrowdPeopleEvent()
        # return EventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa
        cv_bgr2rgb_batch(batches)
        result = self.model(batches)

        alarms = self.alarm_event_manager(result, stream_ids, user_params)

        if len(alarms) > 0:
            return {
                ALARMS_KEY: alarms,
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
        else:
            return None
