from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.khonkaen_peoplecounting.event import PeoplecountingEvent
from pia_prod.AI.modules.khonkaen_peoplecounting.roi_manager import PeoplecountingRoIManager
from queue import Queue
from pia.ai.tasks.CP.base import CPONNXConfig
from pia_prod.AI.modules.khonkaen_peoplecounting.config import CROWD_PEOPLE_ONNX_MODEL_PATH, DEVICE
from pia_prod.AI.modules.khonkaen_peoplecounting.model import ClipEBCOnnxTorch
from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    NUM_OF_OBJECT_KEY,
)


class PeoplecountingService(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        self.is_needed_cvt_color = True

    def _load_model(self):
        config = CPONNXConfig(model_path=CROWD_PEOPLE_ONNX_MODEL_PATH, device=DEVICE)
        # self.model = PiaTorchModel(target_task="CP", target_model="clip_ebc_onnx", config=config)
        self.model = ClipEBCOnnxTorch(config=config)

    def _init_values(self):
        pass

    def _load_event_manager(self):
        return PeoplecountingEvent()

    def _load_roi_manager(self):
        return PeoplecountingRoIManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        if not self.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)

        # roi processing
        cropped_batches, _ = self.roi_manager.process_batches_with_roi(batches, user_params)

        num_of_people_list = []
        for cropped_batch in cropped_batches:
            result = self.model(cropped_batch)
            num_of_people_list.append(int(result[0]))

        if len(num_of_people_list) > 0:
            return {
                ALARMS_KEY: [],
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
                NUM_OF_OBJECT_KEY: num_of_people_list,
            }
        else:
            return None
