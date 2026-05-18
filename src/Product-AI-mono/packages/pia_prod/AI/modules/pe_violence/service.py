from queue import Queue
from collections import defaultdict, deque
import torch
import os

from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.pe_violence.event import PVEventManager
from pia_prod.AI.modules.pe_violence.config import (
    IMG_SIZE,
    DEVICE,
    VIOLENCE_PE_TXT_FEATURE_PATH,
    VIOLENCE_PE_MODEL_TRT_PATH,
    TEMPORAL_SIZE,
    ALARM_QUEUE_SIZE,
    ALARM_THRESHOLD,
)
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image
from pia_prod.AI.modules.pe_violence.roi_manager import PEViolenceRoIManager
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class PVService(ServiceBase):
    """
    [업데이트 버전]
    - 입력: 4프레임 간격의 프레임 (test_pe_violence.py에서 전달)
    - 처리:
        1. 8프레임 모으기
        2. RGB 변환 및 전처리
        3. TensorRT 모델 추론
        4. 프레임 임베딩 평균 풀링 → 비디오 임베딩 생성
        5. 정규화 후 텍스트 임베딩과 유사도 계산
        6. violence 여부 판단 및 EventManager로 알람 전송
    """

    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        self.is_needed_cvt_color = True
        self.frame_buffers = defaultdict(lambda: deque(maxlen=TEMPORAL_SIZE + 1)) # Video Embedding
        self.num_gather_frames = 3  # fps 3
        self.gather_frame_buffers = defaultdict(lambda: deque(maxlen=self.num_gather_frames))
        self.debug = False

    def _init_values(self):
        self.category_txt_vectors = {}

    def _load_model(self):
        self.model = PiaONNXTensorRTModel(VIOLENCE_PE_MODEL_TRT_PATH, device="cuda", half=True)
        self.image_size = IMG_SIZE
        self._load_text_vectors()

    def _load_roi_manager(self):
        return PEViolenceRoIManager()

    def _load_event_manager(self):
        return PVEventManager(
            alarm_queue_size=ALARM_QUEUE_SIZE,
            alarm_threshold=ALARM_THRESHOLD,
        )

    def _stack_text_vectors(self, vec_list: list[torch.Tensor]) -> torch.Tensor:
        """
        vec_list: list of tensors shaped [1, 1024] (or [1024])
        returns: [1024, N] normalized
        """
        # [N, 1, 1024] -> [N, 1024]
        mat = torch.cat([v.squeeze(0) if v.dim() == 2 else v for v in vec_list], dim=0)  # [N,1024]
        mat = mat / mat.norm(dim=-1, keepdim=True)
        return mat.t().contiguous()  # [1024, N]

    def _load_text_vectors(self):
        """normal / violence 기준 텍스트 임베딩 로드 후 정규화"""
        text_vector_classes = sorted(os.listdir(VIOLENCE_PE_TXT_FEATURE_PATH))

        assert text_vector_classes == [
            "normal",
            "violence",
        ], "text vector files must be normal / violence"

        for text_vector_class in text_vector_classes:
            text_vector_path = os.path.join(VIOLENCE_PE_TXT_FEATURE_PATH, text_vector_class)
            text_vector_files = os.listdir(text_vector_path)
            text_vec_list = []
            for text_vector_file in text_vector_files:
                file_path = os.path.join(text_vector_path, text_vector_file)  # FIX: use new var
                text_vector = torch.load(file_path, map_location=DEVICE)
                text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
                # print(f"Load {text_vector_file} text vector: {text_vector.shape}")
                text_vec_list.append(text_vector)
            # print(f"Load {text_vector_class} text vectors: {len(text_vec_list)}")
            stacked_text_vector = torch.cat(text_vec_list, dim=0)  # [N,1024]
            stacked_text_vector = stacked_text_vector.t().contiguous()
            self.category_txt_vectors[text_vector_class] = stacked_text_vector

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        if not self.is_torch_batches(batches, speed_mode=True):
            cv_bgr2rgb_batch(batches)

        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        processed_batches = preprocess_image(
            cropped_batches, size=self.image_size[0], device=DEVICE
        )  # [B, C, H, W]

        # 1) gather_frame_buffers에 프레임 축적 (3프레임 모을 때까지)
        for stream_id, batch in zip(stream_ids, processed_batches):
            self.gather_frame_buffers[stream_id].append(batch)

        user_param_map = {sid: param for sid, param in zip(stream_ids, user_params)}

        # 2) 3프레임 모인 스트림만 인코딩
        encode_stream_ids = []
        encode_tensors = []
        for stream_id in stream_ids:
            gbuf = self.gather_frame_buffers[stream_id]
            if len(gbuf) == self.num_gather_frames:
                encode_tensors.append(torch.stack(list(gbuf)))  # [3, C, H, W]
                gbuf.clear()
                encode_stream_ids.append(stream_id)

        if encode_stream_ids:
            encode_input = torch.stack(encode_tensors)  # [B_enc, 3, C, H, W]

            with torch.inference_mode():
                embeddings = self.model(encode_input)  # [B_enc, 3, 1024]
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            for sid, emb in zip(encode_stream_ids, embeddings):
                self.frame_buffers[sid].extend(emb.unbind(0))  # 3개를 한번에 extend
                # Pop index 1 to keep oldest frame for longer temporal context
                # [0, 1, 2, ..., 8] -> [0, 2, 3, ..., 8]
                if len(self.frame_buffers[sid]) > TEMPORAL_SIZE:
                    del self.frame_buffers[sid][1]

        # 4) 버퍼가 꽉 찬 스트림에 대해 유사도 계산
        ready_stream_ids = []
        video_embeddings = []
        for stream_id in stream_ids:
            buf = self.frame_buffers[stream_id]
            if len(buf) >= TEMPORAL_SIZE:
                stacked = torch.stack(list(buf))  # [8, 1024]
                video_emb = stacked.mean(dim=0)  # [1024]
                video_embeddings.append(video_emb)
                ready_stream_ids.append(stream_id)

        if not ready_stream_ids:
            return

        latest_frames = {}
        for stream_id, frame in zip(stream_ids, batches):
            if stream_id in ready_stream_ids:
                latest_frames[stream_id] = frame

        vis_vectors = torch.stack(video_embeddings)  # [B_ready, 1024]
        vis_vectors = vis_vectors / vis_vectors.norm(dim=-1, keepdim=True)

        # Top-1 Algorithm
        sim_normal = vis_vectors @ self.category_txt_vectors["normal"]
        sim_violence = vis_vectors @ self.category_txt_vectors["violence"]
        results = sim_violence.max(dim=1).values > sim_normal.max(dim=1).values
        results = results.detach().cpu().tolist()

        user_param_list = [user_param_map[sid] for sid in ready_stream_ids]
        frame_list = [latest_frames[sid] for sid in ready_stream_ids]

        alarms = self.alarm_event_manager.update(results, ready_stream_ids)
        if self.debug:
            results = {
                ALARMS_KEY: alarms,
                BATCHES_KEY: batches,
                STREAM_IDS_KEY: stream_ids,
                USER_PARAMS_KEY: user_params,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
            self.send_alarm(results)

        if len(alarms) > 0:
            return {
                ALARMS_KEY: alarms,
                BATCHES_KEY: frame_list,
                STREAM_IDS_KEY: ready_stream_ids,
                USER_PARAMS_KEY: user_param_list,
                IS_NEEDED_CVT_COLOR_KEY: self.is_needed_cvt_color,
            }
        else:
            return None
