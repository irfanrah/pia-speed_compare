import logging
from collections import defaultdict
from typing import Dict, Optional

from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.vehicle_reverse.config import (
    CATEGORY_NAME,
    EVENT_QUEUE_SIZE,
    DETECTION_INPUT_SIZE,
)
from pia_prod.AI.modules.vehicle_reverse.event import VehicleReverseEventManager
from pia_prod.AI.modules.vehicle_reverse.roi_manager import VehicleReverseRoIManager
from pia_prod.AI.modules.vehicle_reverse.processor import WrongWayProcessor
from pia_prod.AI.modules.vehicle_reverse.debug_utils import save_snapshot_for_vehicle_reverse
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)

logger = logging.getLogger(__name__)


class VehicleReverseService(ServiceBase):
    def _init_values(self):
        self.category_name = CATEGORY_NAME
        self.processor: Optional[WrongWayProcessor] = None
        self._model = None
        self._stream_wrong_state: Dict[str, bool] = defaultdict(bool)
        self._stream_miss_counts: Dict[str, int] = defaultdict(int)
        self._state_persist_frames = max(1, EVENT_QUEUE_SIZE)
        self.save_video = False  # Debugging 용도, test_1batch에서 사용
        self.video_manager = None  # Debugging 용도, test_1batch에서 사용
        self.detected_counter = 0  # Debugging 용도, test_1batch에서 사용

    def _load_model(self):
        self.processor = WrongWayProcessor()
        self._model = self.processor

    def _load_roi_manager(self):
        return VehicleReverseRoIManager()

    def _load_event_manager(self):
        return VehicleReverseEventManager()

    def _ensure_processor(self) -> bool:
        if self.processor is None:
            self._load_model()
        return self.processor is not None

    def model_inference(self, batches):
        letterboxed_results = []
        ratio_results = []
        dwdh_results = []

        for image in batches:
            letterboxed, ratio, dwdh = self.processor.tracker._letterbox(
                image, DETECTION_INPUT_SIZE
            )
            letterboxed_results.append(letterboxed)
            ratio_results.append(ratio)
            dwdh_results.append(dwdh)

        results = self.processor.tracker.model(
            letterboxed_results, **self.processor.tracker.predict_kwargs
        )

        return results, ratio_results, dwdh_results

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        stream_flags: Dict[str, bool] = {}

        # Preprocess 및 배치 처리
        inference_results, ratio_results, dwdh_results = self.model_inference(batches)
        rois, direction_vectors = self.roi_manager.get_rois_info(
            batches=batches, user_params=user_params
        )

        events_list = []
        for idx in range(len(batches)):
            now_inference_result = inference_results[idx]
            now_ratio = ratio_results[idx]
            now_dwdh = dwdh_results[idx]
            now_stream_id = stream_ids[idx]
            now_image = batches[idx]

            now_roi = rois[idx]
            now_direction_vector = direction_vectors[idx]

            events = self.processor.process_frame(
                im0=now_image,
                frame_idx=self.frame_cnt,
                stream_id=now_stream_id,
                now_inference_result=now_inference_result,
                now_ratio=now_ratio,
                now_dwdh=now_dwdh,
                now_roi=now_roi,
                now_direction_vector=now_direction_vector,
            )
            if events:
                has_wrong = any(evt.get("is_wrong") for evt in events)
                self._stream_wrong_state[now_stream_id] = has_wrong
                self._stream_miss_counts[now_stream_id] = 0
            else:
                miss_cnt = self._stream_miss_counts[now_stream_id] + 1
                self._stream_miss_counts[now_stream_id] = miss_cnt
                if miss_cnt >= self._state_persist_frames:
                    self._stream_wrong_state[now_stream_id] = False
            stream_flags[now_stream_id] = self._stream_wrong_state[now_stream_id]
            events_list.append(events)

        if stream_flags and self.alarm_event_manager is not None:
            per_stream_events = [
                {"stream_id": sid, "is_wrong": flag} for sid, flag in stream_flags.items()
            ]
            alarms = self.alarm_event_manager.update(per_stream_events)

        if self.logging_flag:
            save_snapshot_for_vehicle_reverse(
                images=batches,
                stream_ids=stream_ids,
                events_list=events_list,
                video_mode=self.save_video,
                video_instance=self.video_manager,
            )

        self.frame_cnt += 1

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
