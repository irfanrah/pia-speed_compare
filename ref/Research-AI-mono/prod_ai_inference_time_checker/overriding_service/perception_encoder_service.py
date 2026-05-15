from pia_prod.AI.modules.perception_encoder.service import PEService
from queue import Queue
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image
from pia.vision.preprocessing import cv_bgr2rgb_batch
from pia_prod.AI.modules.perception_encoder.config import (
    TEMPORAL_SIZE,
)
import time


class ProfilePEService(PEService):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)
        # ======= 결과 정리 ======
        self.stages = ["total", "model_preprocess", "model_inference", "postprocess_logic", "send_alarm"]

    def is_sink(self) -> bool:
        ts = [self.t0, self.t1, self.t2, self.t3, self.t4]
        for i in range(1, len(ts)):
            if ts[i] < ts[i - 1]:
                # print(f"⚠️ Timestamp order reversed: t{i-1}={ts[i-1]:.6f}, t{i}={ts[i]:.6f}")
                return False
        return True

    def _detect(self, **datas):
        self.t0 = time.perf_counter()  # start
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        # target_txt_vector, target_sentences = self.txtvec_manager.update(
        #   stream_ids, user_params) # FIX : prompts 고정으로 하기로 함

        # batch encode
        # sequnce length 1로 고정
        cv_bgr2rgb_batch(batches)
        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        image_cuda = preprocess_image(cropped_batches)
        self.t1 = time.perf_counter()  # model_preprocess end
        # torch_batches = torch.from_numpy(np.stack(batches, axis=0)).to(DEVICE)
        visual_vectors = self.model(image_cuda)
        self.t2 = time.perf_counter()  # model_inference end
        for stream_id, visual_vector in zip(stream_ids, visual_vectors):
            # FIX : image 기반으로 처리하기로 함
            while self.stream_vector_queues[stream_id].__len__() < TEMPORAL_SIZE:  # temporal size 1 로 설정함
                self.stream_vector_queues[stream_id].append(self.zero_mask_vec)
            self.stream_vector_queues[stream_id].append(visual_vector)  # TODO : 연결 끊긴 스트림은 삭제 해야함

        alarms, predict_infos = self.alarm_event_manager(self.stream_vector_queues, stream_ids, user_params)
        self.t3 = time.perf_counter()  # postprocess_logic end
        self.send_alarm(alarms, batches, stream_ids, user_params, is_needed_cvt_color=True)
        self.t4 = time.perf_counter()  # send_alarm end

        times = [
            (self.t4 - self.t0) * 1000,  # total
            (self.t1 - self.t0) * 1000,  # model_preprocess
            (self.t2 - self.t1) * 1000,  # model_inference
            (self.t3 - self.t2) * 1000,  # postprocess_logic
            (self.t4 - self.t3) * 1000,  # send_alarm
        ]

        if self.is_sink():
            self.time_dict["total"].append(times[0])
            self.time_dict["model_preprocess"].append(times[1])
            self.time_dict["model_inference"].append(times[2])
            self.time_dict["postprocess_logic"].append(times[3])
            self.time_dict["send_alarm"].append(times[4])

        return alarms, batches, stream_ids, user_params, self.is_needed_cvt_color
