import torch
import os
import numpy as np
from queue import Queue
from collections import defaultdict, deque

from pia_prod.AI.modules.pe_vle_2stage_sync.config import (
    TWOSTAGE_PE_QUEUE_SIZE,
    TWOSTAGE_PE_ALARM_DURATION_THRESHOLD,
    QWEN3VLE_TEMPORAL_SIZE,
    TWOSTAGE_TO_PE_CATEGORY_EVENT_MAP,
    TWOSTAGE_TO_VLE_CATEGORY_EVENT_MAP,
    PE_TO_TWOSTAGE_CATEGORY_EVENT_MAP,
)
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.global_config import (
    TEAM_KEY,
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
    USER_PARAM_KEY,
    RET_EVENT_KEY,
)
from pia_prod.AI import PEService
from pia_prod.AI import Qwen3VLETrtService


class PeVle2StageSyncService(ServiceBase):
    """
    2-Stage Anomaly Detection Service:
 
    Flow:
        1. Every frame is passed through PE (queue_size=10, threshold=5).
        2. Original frames are continuously buffered for Qwen3's temporal window.
        3. When PE raises a new anomaly alarm (status=1), Qwen3 verifies it:
           - Anomaly confirmed → PE alarm stands → Final Alarm raised.
           - Returns "normal"  → PE queue entry flipped to 0, status reset to 0.
    """
    def __init__(self, analysis_data_queue: Queue):
        self.pe_service = None
        self.qwen_service = None

        self.pe_queue_size = TWOSTAGE_PE_QUEUE_SIZE
        self.pe_alarm_duration_threshold = TWOSTAGE_PE_ALARM_DURATION_THRESHOLD
        self.qwen3vle_temporal_size = QWEN3VLE_TEMPORAL_SIZE

        # Buffer to store original frames that will be fed to Qwen3
        self.original_frame_buffers = defaultdict(
            lambda: deque(maxlen=self.qwen3vle_temporal_size)
        )

        super().__init__(analysis_data_queue)

    def _init_values(self):        
        return

    def _load_model(self):
        # Temporarily unset TEAM so child services don't start their own threads.
        # The parent service's thread handles the queue.
        saved_team = os.environ.pop(TEAM_KEY, None)

        if self.pe_service is None:
            self.pe_service = PEService(self.analysis_data_queue)
        if self.qwen_service is None:
            self.qwen_service = Qwen3VLETrtService(self.analysis_data_queue)
        
        # Set the TEAM again for the two-stage service
        if saved_team is not None:
            os.environ["TEAM"] = saved_team

        self._configure_services()

    def _load_event_manager(self):
        return

    def _configure_services(self):
        """Override PE and Qwen3 configs to match the Two-Stage flow."""
        # PE Config
        self.pe_service.alarm_event_manager.duration_queue = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.pe_queue_size))
        )
        self.pe_service.alarm_event_manager.alarm_duration = self.pe_alarm_duration_threshold

        # Qwen3VLE Config (Only reload if temporal size is different)
        if self.qwen_service.temporal_size != self.qwen3vle_temporal_size:
            self.qwen_service.temporal_size = self.qwen3vle_temporal_size
            self.qwen_service.frame_buffers = defaultdict(
                lambda: deque(maxlen=self.qwen3vle_temporal_size)
            )

            # Free the previous model from GPU before reloading
            del self.qwen_service.model
            torch.cuda.empty_cache()

            self.qwen_service._load_model()

    # -------------------------------------------------------------------------
    # Category Remapping
    # -------------------------------------------------------------------------
    @staticmethod
    def _remap_user_params(user_params, category_map):
        """Create a shallow copy of user_params with retEvent categories remapped."""
        remapped = []
        for param in user_params:
            new_param = {**param}
            if USER_PARAM_KEY in new_param:
                new_param[USER_PARAM_KEY] = {**new_param[USER_PARAM_KEY]}
                ret_events = new_param[USER_PARAM_KEY].get(RET_EVENT_KEY, {})
                new_param[USER_PARAM_KEY][RET_EVENT_KEY] = {
                    category_map.get(key, key): value
                    for key, value in ret_events.items()
                }
            remapped.append(new_param)
        return remapped

    @staticmethod
    def _remap_alarm_categories(alarms, category_map):
        """Remap alarm category IDs in-place using the given map."""
        for sid in alarms:
            old_cat = alarms[sid][1]
            alarms[sid][1] = category_map.get(old_cat, old_cat)

    # -------------------------------------------------------------------------
    # Helper Function
    # -------------------------------------------------------------------------
    def _run_stage1_pe(self, datas):
        """
        Run PE preprocess → model → predict → update alarm status.
        Returns pe_result if there is an alarm (status=1 or status=3), else None.
        """
        pe_result = self.pe_service._detect(**datas)
        if not pe_result or ALARMS_KEY not in pe_result:
            return None
        return pe_result
    
    def _prepare_qwen_inputs(self, alarmed_sids, sid_to_param):
        """
        Build batched inputs for Qwen3 from buffered frames of alarmed streams.
        Skips streams with insufficient buffered frames (PE alarm stands for those).
        Returns (qwen_batches, qwen_sids, qwen_params).
        """
        qwen_batches, qwen_sids, qwen_params = [], [], []

        for sid in alarmed_sids:
            buffered_frames = list(self.original_frame_buffers[sid])
            if len(buffered_frames) < self.qwen3vle_temporal_size:
                continue  # Not enough frames yet — PE alarm stands unverified

            self.qwen_service.frame_buffers[sid].clear()

            for frame in buffered_frames:
                qwen_batches.append(frame)
                qwen_sids.append(sid)
                qwen_params.append(sid_to_param[sid])

        return qwen_batches, qwen_sids, qwen_params


    def _correct_pe_false_alarm(self, qwen_result, pe_result):
        """
        For each stream where Qwen3 returned "normal" (all anomaly flags are False):
        - Flip the last PE queue entry from 1 → 0.
        - Reset PE alarm status to 0 (no_event).
        - Remove the stream from the alarm dict.
        """
        pe_em = self.pe_service.alarm_event_manager

        for sid, pred in zip(qwen_result["stream_ids"], qwen_result["predictions"]):
            # Check if prediction is "normal" (e.g., neither fire nor smoke is True)
            # any(pred.values()) is True if at least one value is True.
            # So, not any(...) means it is completely normal.
            is_normal = not any(pred.values())

            if not is_normal or sid not in pe_result[ALARMS_KEY]:
                continue

            category_id = pe_result[ALARMS_KEY][sid][1]

            pe_queue = pe_em.duration_queue[sid][category_id]
            if pe_queue and pe_queue[-1]:
                pe_queue[-1] = 0

            pe_em.event_status[sid][category_id] = 0
            pe_result[ALARMS_KEY].pop(sid)

    def _run_stage2_qwen3vle(self, alarmed_sids, pe_result, stream_ids, user_params):
        """
        Verify PE alarms with Qwen3 and correct false alarms:
        1. Prepare buffered frames for Qwen3 inference.
        2. Run Qwen3 inference on the buffered frames.
        3. For streams where Qwen3 returns "normal" (false alarm),
           correct PE's prediction queue, event status, and remove the alarm.
        Returns the (possibly pruned) pe_result, or None if all alarms were corrected.
        """
        sid_to_param = dict(zip(stream_ids, user_params))

        qwen_batches, qwen_sids, qwen_params = self._prepare_qwen_inputs(
            alarmed_sids, sid_to_param
        )
        if not qwen_sids:
            return pe_result # No streams had enough buffered frames for Qwen3

        qwen_result = self.qwen_service._predict(
            batches=qwen_batches,
            stream_ids=qwen_sids,
            user_params=qwen_params,
        )
        if not qwen_result:
            return pe_result

        # Correct PE's prediction queue, event status, and remove the alarm
        self._correct_pe_false_alarm(qwen_result, pe_result)

        return pe_result if pe_result[ALARMS_KEY] else None

    # -------------------------------------------------------------------------
    # Main detect
    # -------------------------------------------------------------------------
    def _detect(self, **datas):
        stream_ids = datas["stream_ids"]
        batches = datas["batches"]
        user_params = datas["user_params"]  # Original two-stage params

        # Remap categories for each child service
        pe_datas = {**datas, "user_params": self._remap_user_params(user_params, TWOSTAGE_TO_PE_CATEGORY_EVENT_MAP)}
        vle_user_params = self._remap_user_params(user_params, TWOSTAGE_TO_VLE_CATEGORY_EVENT_MAP)

        # Stage 1: PE pipeline (cv_bgr2rgb_batch mutates batches in-place)
        pe_result = self._run_stage1_pe(pe_datas)

        # Buffer the converted (RGB) frames for Qwen3VLE
        for sid, batch in zip(stream_ids, batches):
            self.original_frame_buffers[sid].append(
                np.copy(batch) if isinstance(batch, np.ndarray) else batch
            )
        if pe_result is None:
            return None

        # Only newly started alarms (status=1 / True) need Qwen3 verification.
        # Alarm endings (status=3 / False) pass through directly.
        alarmed_sids = [
            sid for sid, alarm_info in pe_result[ALARMS_KEY].items()
            if alarm_info[0] is True
        ]
        if not alarmed_sids:
            # Remap PE alarm categories back to two-stage names before returning
            self._remap_alarm_categories(pe_result[ALARMS_KEY], PE_TO_TWOSTAGE_CATEGORY_EVENT_MAP)
            return pe_result

        # Stage 2: Verify alarmed streams with Qwen3 and correct false alarms
        final_result = self._run_stage2_qwen3vle(alarmed_sids, pe_result, stream_ids, vle_user_params)
        if final_result is None:
            return None

        # Remap PE alarm categories back to two-stage names
        self._remap_alarm_categories(final_result[ALARMS_KEY], PE_TO_TWOSTAGE_CATEGORY_EVENT_MAP)

        # Build final output for confirmed alarm streams
        alarm_sids = list(final_result[ALARMS_KEY].keys())
        frame_map = dict(zip(stream_ids, batches))
        param_map = dict(zip(stream_ids, user_params))

        return {
            ALARMS_KEY: final_result[ALARMS_KEY],
            BATCHES_KEY: [frame_map[sid] for sid in alarm_sids],
            STREAM_IDS_KEY: alarm_sids,
            USER_PARAMS_KEY: [param_map[sid] for sid in alarm_sids],
            IS_NEEDED_CVT_COLOR_KEY: True,
        }