from pia_prod.AI.bases.service_base import ServiceBase
from queue import Queue
from pia.ai.tasks.VQA.base import VQAConfig
from pia.ai.model import PiaTorchModel
from pia_prod.AI.modules.samsung_internvl3.event import VQAEventManager
from pia_prod.AI.modules.samsung_internvl3.config import (
    INTERNVL3_MODEL_HF_PATH,
    N_GPUS,
    FRAME_PER_TILE_MAX_NUM,
    DEVICE,
    MODEL_INF_DATA_TYPE,
    FIRE_CATEGORY,
    FALLDOWN_CATEGORY,
    MAX_NEW_TOKEN,
)
from pia_prod.AI.modules.samsung_internvl3.prompts import FIRE, FALLDOWN
from pia_prod.AI.global_config import USER_PARAM_KEY, VQA_EVENT_KEY
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class InternVL3Service(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)

    def _load_model(self):
        self.config = VQAConfig(
            model_path=INTERNVL3_MODEL_HF_PATH,
            device=DEVICE,
            frame_per_tile_max_num=FRAME_PER_TILE_MAX_NUM,
            data_type=MODEL_INF_DATA_TYPE,
            n_gpus=N_GPUS,
            save_cache_dir=INTERNVL3_MODEL_HF_PATH,
            max_new_tokens=MAX_NEW_TOKEN,
        )
        self.model = PiaTorchModel(target_task="VQA", target_model="internVL3", config=self.config)

    def _init_values(self):
        pass

    def _load_event_manager(self):
        return VQAEventManager()

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
                if category in FIRE_CATEGORY:
                    prompts.append(FIRE)
                elif category in FALLDOWN_CATEGORY:
                    prompts.append(FALLDOWN)
                else:
                    continue
            categories.append(temp_categories)
        return re_batches, prompts, categories

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        # category 별로 prompts 선택 및 batch 늘리기
        batches, prompts, categories = self._match_category2prompt(batches, user_params)

        responses = self.model(batches, prompts)

        alarms = self.alarm_event_manager.update(responses, categories, stream_ids)

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
