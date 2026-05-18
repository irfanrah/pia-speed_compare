from pia_prod.AI.bases.service_base import ServiceBase
from queue import Queue
from collections import defaultdict
from pia_prod.AI.modules.pe_distribution.event import PEDistributionEventManager
from pia_prod.AI.modules.pe_distribution.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_distribution.prompts import load_text_feature  # PROMPTS
from .trt_utils import preprocess_image
from pia.vision.preprocessing import cv_bgr2rgb_batch
from collections import deque
from pia_prod.AI.modules.pe_distribution.config import (
    IMG_SIZE,
    DEVICE,
    TEMPORAL_SIZE,
    PE_DISTRIBUTION_TXT_FEATURE_PATH,
    PE_DISTRIBUTION_TRT_PATH,
)
import torch
from .trt_load import TRTInference
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class PEDistributionService(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        self.alarm_event_manager.prepare_vectors_on_gpu(self.category_txt_vectors)
        self.is_needed_cvt_color = True

    def _load_model(self):
        self.model = TRTInference(PE_DISTRIBUTION_TRT_PATH)
        self.image_size = IMG_SIZE
        self._init_default_values()

    def _init_default_values(self):
        # 추상 패턴에서 init_values보다 load_model이 늦게 로드 되어서 여기에 삽입함
        # FIX : prompts 고정으로 하기로 함
        # self.txtvec_manager = TxtvecManager(model=self.model, device=DEVICE)
        # input_txt = [p for v in PROMPTS.values() for p in v ]
        # self.txt_vectors = self.model(text=input_txt)
        self._get_txt_vector_group_by_category()
        zero_mask_vec = self.model(
            torch.zeros(
                size=(1, 3, self.image_size[0], self.image_size[1]),
                dtype=torch.float32,
                device=DEVICE,
            )
        )
        self.zero_mask_vec = zero_mask_vec / zero_mask_vec.norm(dim=-1)

    def _load_roi_manager(self):
        return PERoIManager()

    def _get_txt_vector_group_by_category(self):
        ID_list, class_list, prompt_list, text_features = load_text_feature(
            PE_DISTRIBUTION_TXT_FEATURE_PATH, DEVICE
        )
        self.category_txt_vectors['ids'] = ID_list
        self.category_txt_vectors['vectors'] = text_features
        self.category_txt_vectors['class_list'] = class_list
        self.category_txt_vectors['prompt_list'] = prompt_list

    def _init_values(self):
        self.category_txt_vectors = defaultdict()
        self.stream_vector_queues = defaultdict(lambda: deque(maxlen=TEMPORAL_SIZE))

    def _load_event_manager(self):
        return PEDistributionEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        if not self.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)

        # target_txt_vector, target_sentences = self.txtvec_manager.update(
        #   stream_ids, user_params) # FIX : prompts 고정으로 하기로 함

        # batch encode
        # sequnce length 1로 고정
        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        image_cuda = preprocess_image(cropped_batches)
        # torch_batches = torch.from_numpy(np.stack(batches, axis=0)).to(DEVICE)
        visual_vectors = self.model(image_cuda)
        for stream_id, visual_vector in zip(stream_ids, visual_vectors):
            # FIX : image 기반으로 처리하기로 함
            while (
                self.stream_vector_queues[stream_id].__len__() < TEMPORAL_SIZE
            ):  # temporal size 1 로 설정함
                self.stream_vector_queues[stream_id].append(self.zero_mask_vec)
            self.stream_vector_queues[stream_id].append(
                visual_vector
            )  # TODO : 연결 끊긴 스트림은 삭제 해야함

        alarms, predict_infos = self.alarm_event_manager(
            self.stream_vector_queues, stream_ids, user_params
        )

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
