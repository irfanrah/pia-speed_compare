from typing import List, Dict, Tuple
import numpy as np
import torch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.aramco_loitering.config import (
    DEVICE,
    PERSON_DETECTION_MODEL_TRT_PATH,
    OD_TARGET_CLASSES,
    OD_INPUT_SIZE,
    LIMITED_NUM_OF_CAMERA,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_NMS_THRESHOLD,
    OD_CONFIDENCE_THRESHOLD,
    TRACKER_DICT,
)

from pia_prod.AI.modules.aramco_loitering.event import LoiteringEventManager
from pia_prod.AI.modules.aramco_loitering.roi_manager import LoiteringRoIManager
from pia.vision.preprocessing.resize import LetterBoxTorch
from pia_prod.AI.modules.aramco_loitering.postprocess import torch_non_max_suppression
from pia.vision.postprocessing.bbox import modified2origin_coordinate
from pia_prod.AI.modules.aramco_loitering.debug_utils import save_snapshot_for_odtracker
from pia_prod.AI.utils.utils import free_autobackend
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.modules.vanguard_patient.tracker import MultiStreamPiaOCSort
from pia_prod.AI.modules.vanguard_patient.preprocess import (
    is_inside_roi,
    foot_point,
)
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class LoiteringService(ServiceBase):
    ZERO_TENSOR = torch.tensor([0], device=DEVICE)
    TRUE_TENSOR = torch.tensor(True, device=DEVICE)

    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.category_name = "loitering"
        self.save_video = False  # Debugging 용도
        self.video_manager = None  # Debugging 용도
        self.detected_counter = 0  # Debugging 용도

    def __del__(self):
        self.del_model()

    def _init_values(self):
        self.od_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA, target_size=OD_INPUT_SIZE, device=DEVICE
        )
        self._load_tracker()

    def _load_roi_manager(self):
        return LoiteringRoIManager()

    def _load_model(self):
        self.object_detection_instance = PiaONNXTensorRTModel(
            model_path=PERSON_DETECTION_MODEL_TRT_PATH, device=DEVICE
        )

    def _load_tracker(self):
        self.tracker = MultiStreamPiaOCSort(
            det_thresh=TRACKER_DICT["det_thresh"],
            max_objs=LIMITED_NUM_OF_PERSON_PER_CAMERA,
            max_age=TRACKER_DICT["max_age"],
            min_hits=TRACKER_DICT["min_hits"],
            iou_threshold=TRACKER_DICT["iou_threshold"],
            use_byte=TRACKER_DICT["use_byte"],
        )

    def del_model(self):
        """모델을 삭제합니다."""
        if hasattr(self, "object_detection_instance"):
            free_autobackend(self.object_detection_instance)

        if hasattr(self, "classification_instance"):
            free_autobackend(self.classification_instance)

    def _load_event_manager(self):
        return LoiteringEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        # PGIE 결과 받아오기
        tracked_nms_od_result, cropped_batches, categories_batches, track_info = (
            self.get_pgie_results(batches=batches, stream_ids=stream_ids, user_params=user_params)
        )

        # 알람 결과 받아오기
        alarms = self.get_alarm_loitering(track_info, stream_ids)

        # ---------------------------------------------------------
        # [신규] Logging Flag ON 일 때만 시각화 관련 최소 연산 수행
        # ---------------------------------------------------------
        if getattr(self, "logging_flag", False):
            # cropped_batches 길이에 맞춰서 리스트 초기화
            total_bbox_info = [[] for _ in range(len(cropped_batches))]

            # 각각의 crop된 이미지가 어떤 stream_id를 가지는지 매핑
            current_crop_stream_ids = [stream_ids[idx] for idx in categories_batches]

            for cropped_batch_idx, batch_idx_info in enumerate(categories_batches):
                track_results = tracked_nms_od_result[cropped_batch_idx]

                if track_results is None or len(track_results) == 0:
                    continue

                for trk in track_results:
                    # [x1, y1, x2, y2, track_id]
                    x1, y1, x2, y2, track_id = map(int, trk[:5])

                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    # Letterbox(모델 입력 크기) -> Crop 원본 크기로 좌표 원복
                    origin_bbox = modified2origin_coordinate(
                        xyxy=(x1, y1, x2, y2),
                        now_shape=OD_INPUT_SIZE,
                        original_shape=cropped_batches[cropped_batch_idx].shape[1:3],
                    )

                    # id 정보 추가 (시각화를 위해)
                    origin_bbox.append(track_id)
                    total_bbox_info[cropped_batch_idx].append(origin_bbox)

            # 디버그 그리기 함수 호출
            save_snapshot_for_odtracker(
                images=cropped_batches,  # Crop된 이미지들
                stream_ids=current_crop_stream_ids,  # 매핑된 카메라 ID들
                total_bbox_info=total_bbox_info,  # 원복된 bbox 정보들
                event_manager=self.alarm_event_manager,  # 점수 및 알람 상태 조회를 위해 통째로 넘김
                roi_dict=self.roi_manager.roi_dict,  # ROI 그리기 용도
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

    def get_pgie_results(
        self,
        batches: List[np.array],
        stream_ids: List[str],
        user_params: List,
    ) -> Tuple[List[np.array], List[np.array], List[int], Dict[str, List[int]]]:
        """N개로 들어오는 이미지를 받아 object detection task를 수행하고, 결과를 반환합니다.

        Args:
            batches (List[np.array]): len(batches)개만큼 들어오는 이미지
            stream_ids (list[str]): 카메라의 stream_ids
            user_params (list): AddStreamModel 데이터클래스에 정의된 파라미터들. ROI 정의시 사용됨

        Returns:
            tracked_nms_od_result (List[np.array]): NMS 적용 후 tracking까지 완료된 object detection 결과 리스트
            cropped_batches (List[np.array]): ROI 내에서 검출된 object를 crop한 이미지 배치 리스트
            categories_batches (List[int]): 각 crop이 속한 카테고리 인덱스 리스트
            track_info (Dict[str, List[int]]): 각 stream_id별로 검출/추적된 object의 track id 리스트
            예시:

                categories_batches
                    >>> [0, 1, 0, 0]  # 각 crop 이미지가 속한 ROI index(또는 카테고리)를 나타냄. len(batches)보다 클 수 있음

                divided_roi_info[0][2]
                    >>> array([[456, 411],
                        [458, 411],
                        [633, 419],
                        [612, 205],
                        [465, 175]], dtype=int32)
                # 이 예시에서 divided_roi_info[0][2]는 0번 스트림의 세 번째 ROI 다각형 좌표이며,
                # loitering 영역 안에서 bbox(사람 검출 결과)가 ROI 안에 포함되어 있는지 확인하는 데 사용된다.
                    [465, 175]], dtype=int32)
        """
        # 저채도(low S) 영역을 더 저채도로 만들고, 동시에 밝기를 올려 “하늘색(저채도/고명도)”과 “진한 파랑(고채도)”의 시각적 분리를 강화가 목적

        # Preprocess 1 - Crop with RoI
        cropped_batches, categories_batches = self.roi_manager.process_batches_with_roi(
            batches=batches, stream_ids=stream_ids, user_params=user_params
        )
        # Preprocess 2 - Letterbox & Normalize
        resized_images = self.od_letterbox_instance(imgs=cropped_batches)
        preprocessed_images = resized_images / 255.0

        # Inference model
        raw_od_results = self.object_detection_instance(preprocessed_images)  # len() = N
        # Postprocess
        # TODO: conf_thres, iou_thres 리스트로 유저에 의해 변경될 수 있도록 변경
        nms_od_result = torch_non_max_suppression(
            prediction=raw_od_results,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=OD_TARGET_CLASSES,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )
        # [수정 1] Tracker 입력을 위한 stream_ids 확장
        # cropped_batches는 하나의 카메라 이미지가 여러 개로 쪼개졌을 수 있으므로,
        # categories_batches 정보를 이용해 해당 crop이 어떤 stream_id인지 매핑 리스트를 만듭니다.
        # 예: categories_batches가 [0, 0, 1] 이면 -> ['cam1', 'cam1', 'cam2']
        current_crop_stream_ids = [stream_ids[idx] for idx in categories_batches]

        # [수정 2] Tracking 수행
        # tracked_nms_od_result 구조: List[np.array([[x1, y1, x2, y2, track_id], ...])]
        tracked_nms_od_result, track_info = self.tracker.update(
            results=nms_od_result,
            stream_ids=current_crop_stream_ids,  # 확장된 ID 리스트 사용
            img_infos=[OD_INPUT_SIZE] * len(cropped_batches),
            img_sizes=None,
        )
        tracked_nms_od_result, track_info = self.filter_tracks_by_roi(
            tracked_results=tracked_nms_od_result, current_stream_ids=current_crop_stream_ids
        )

        return (
            tracked_nms_od_result,
            cropped_batches,
            categories_batches,
            track_info,
        )

    def get_alarm_loitering(self, track_info: Dict[str, List[int]], stream_ids) -> Dict[str, list]:
        """
        Args:
            tracked_nms_od_result: {'1_pia': [False, True], ...} 또는 [tensor([0])] (감지 없음)
            track_info: {'1_pia': [3, 2], ...}

        Returns:
            alarms: {'1_pia': [True, None]} 형태의 딕셔너리
        """

        # [예외 처리] tracked_nms_od_result 딕셔너리가 아닌 경우 (예: [tensor([0])])
        # 감지된 객체가 없으므로 update 루프를 돌 수 없습니다.
        detected_streams = set()
        if isinstance(track_info, dict):

            # 1. 모든 카메라의 데이터를 EventManager에 업데이트
            for stream_id, track_ids in track_info.items():
                # 상태 갱신
                self.alarm_event_manager.update(
                    results=True, stream_id=stream_id, track_ids=track_ids
                )
                detected_streams.add(stream_id)

        # 화면에 객체가 감지되지 않은 카메라 처리
        for stream_id in stream_ids:
            if stream_id not in detected_streams:
                self.alarm_event_manager.update(results=False, stream_id=stream_id, track_ids=[])

        # 2. 최종 알람 확인 (모든 카메라 체크)
        final_alarms = self.alarm_event_manager.get_alarm()

        return final_alarms

    def filter_tracks_by_roi(self, tracked_results, current_stream_ids):
        """
        Tracking 결과 중, foot_point가 ROI 내부에 있는 객체만 남깁니다.

        Args:
            tracked_results: Tracker 출력 리스트 [Array(N, 5), Array(M, 5), ...]
            current_stream_ids: 확장된 스트림 ID 리스트 (tracked_results와 길이 동일)

        Returns:
            filtered_results: 필터링된 Tracker 출력 리스트
            filtered_track_info: 재구성된 {stream_id: [id1, id2...]} 딕셔너리
        """
        filtered_results = []
        filtered_track_info = {}

        for i, (detections, stream_id) in enumerate(zip(tracked_results, current_stream_ids)):

            # 1. 예외 처리: 탐지된 객체가 없으면 빈 배열 유지
            if detections is None or len(detections) == 0:
                filtered_results.append(np.empty((0, 5)))
                continue

            # 2. ROI 가져오기
            # after_letterbox_calc_origin_roi 키 사용
            roi_poly = self.roi_manager.roi_dict[stream_id].get("after_letterbox_calc_origin_roi")

            # ROI가 없거나 비어있으면 필터링 없이 전체 통과 (또는 정책에 따라 다 삭제)
            if roi_poly is None or len(roi_poly) == 0:
                filtered_results.append(detections)
                # Track info 업데이트
                current_ids = detections[:, 4].astype(int).tolist()
                if stream_id not in filtered_track_info:
                    filtered_track_info[stream_id] = current_ids
                else:
                    filtered_track_info[stream_id].extend(current_ids)
                continue

            # 3. 객체별 ROI 포함 여부 검사
            valid_indices = []
            for idx, bbox in enumerate(detections):
                # bbox: [x1, y1, x2, y2, id]
                fp = foot_point(bbox[:4])

                # ROI 내부인지 확인
                if is_inside_roi(roi_poly, fp):
                    valid_indices.append(idx)

            # 4. 필터링된 결과 생성
            if len(valid_indices) > 0:
                # 유효한 인덱스만 슬라이싱
                kept_detections = detections[valid_indices]
                filtered_results.append(kept_detections)

                # Track Info 재구성 (살아남은 ID만)
                current_ids = kept_detections[:, 4].astype(int).tolist()
                if stream_id not in filtered_track_info:
                    filtered_track_info[stream_id] = current_ids
                else:
                    filtered_track_info[stream_id].extend(current_ids)
            else:
                # 유효한 객체가 하나도 없으면 빈 배열
                filtered_results.append(np.empty((0, 5)))

        return filtered_results, filtered_track_info
