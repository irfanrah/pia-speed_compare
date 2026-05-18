from typing import List, Dict, Tuple
import numpy as np
import torch
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.vanguard_patient.config import (
    DEVICE,
    MODEL_PATIENT_DETECTION_TRT_PATH,
    MODEL_CLOTHCOLOR_CLS_TRT_PATH,
    OD_TARGET_CLASSES,
    OD_INPUT_SIZE,
    CLS_INPUT_SIZE,
    TARGET_CATEGORY_INDEX,
    CLS_CONFIDENCE_THRESHOLD,
    LIMITED_NUM_OF_CAMERA,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_NMS_THRESHOLD,
    OD_CONFIDENCE_THRESHOLD,
    TRACKER_DICT,
)

from pia_prod.AI.modules.vanguard_patient.event import PatientEventManager
from pia_prod.AI.modules.vanguard_patient.roi_manager import PatientRoIManager
from pia.vision.preprocessing.resize import LetterBoxTorch

# from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia_prod.AI.modules.vanguard_patient.postprocess import torch_non_max_suppression
from pia_prod.AI.utils.utils import multi_category_threshold_check
from pia_prod.AI.modules.vanguard_patient.debug_utils import save_snapshot_for_odtrackercls
from pia.vision.postprocessing.bbox import modified2origin_coordinate
from pia_prod.AI.utils.utils import free_autobackend
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.modules.vanguard_patient.tracker import MultiStreamPiaOCSort
from pia_prod.AI.modules.vanguard_patient.preprocess import (
    is_inside_roi,
    foot_point,
    enhance_sky_batch_torch,
)
from pia_prod.AI.global_config import (
    ALARMS_KEY,
    BATCHES_KEY,
    STREAM_IDS_KEY,
    USER_PARAMS_KEY,
    IS_NEEDED_CVT_COLOR_KEY,
)


class PatientService(ServiceBase):
    ZERO_TENSOR = torch.tensor([0], device=DEVICE)
    TRUE_TENSOR = torch.tensor(True, device=DEVICE)
    CLS_CONFIDENCE_THRESHOLD_TENSOR = torch.tensor(CLS_CONFIDENCE_THRESHOLD, device=DEVICE)

    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        self.category_name = "patient"
        self.save_video = False  # Debugging 용도
        self.video_manager = None  # Debugging 용도
        self.detected_counter = 0  # Debugging 용도

    def __del__(self):
        self.del_model()

    def _init_values(self):
        self.od_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA, target_size=OD_INPUT_SIZE, device=DEVICE
        )
        self.cls_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA * LIMITED_NUM_OF_PERSON_PER_CAMERA,
            target_size=CLS_INPUT_SIZE,
            device=DEVICE,
        )
        self._load_tracker()

    def _load_roi_manager(self):
        return PatientRoIManager()

    def _load_model(self):
        self.object_detection_instance = PiaONNXTensorRTModel(
            model_path=MODEL_PATIENT_DETECTION_TRT_PATH, device=DEVICE
        )
        self.classification_instance = PiaONNXTensorRTModel(
            model_path=MODEL_CLOTHCOLOR_CLS_TRT_PATH, device=DEVICE
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
        return PatientEventManager()

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]

        (
            cropped_images,
            matched_info,
            total_bbox_info,
            cropped_batches,
            categories_batches,
            track_info,
        ) = self.get_pgie_results(batches=batches, stream_ids=stream_ids, user_params=user_params)
        # cls는 병렬처리이므로 for문 밖에서 처리
        cls_results, raw_cls_results = self.get_classification_result(
            cropped_images, matched_info
        )  # Classification
        alarms = self.get_alarm_patient(
            cls_results, track_info, stream_ids
        )  # Event Manager, Get Alarm
        if self.logging_flag:
            save_snapshot_for_odtrackercls(
                images=cropped_batches,  # 원본 배치 이미지
                stream_ids=stream_ids,  # 카메라 ID 리스트
                total_bbox_info=total_bbox_info,  # 좌표 및 ID 정보
                raw_cls_results=raw_cls_results,  # Cls 모델 출력값
                matched_info=matched_info,  # 카메라별 객체 수
                category_index=TARGET_CATEGORY_INDEX,  # 타겟 클래스 (tuple/int)
                event_result=self.alarm_event_manager.event_status,  # 이벤트 매니저 상태
                roi_dict=self.roi_manager.roi_dict,
                video_mode=self.save_video,  # (옵션)
                video_instance=self.video_manager,  # (옵션)
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
    ) -> Tuple[List[np.array], Dict[str, int]]:
        """N개로 들어오는 이미지를 받아 object detection task를 수행하고, 결과를 반환합니다.

        Args:
            batches (List[np.array]): len(batches)개만큼 들어오는 이미지
            stream_ids (list[str]): 카메라의 stream_ids
            user_params (dict): AddStreamModel 데이터클래스에 정의된 파라미터들. ROI 정의시 사용됨

        Returns:
            crop_list(List[np.array]): detected된 object를 crop한 이미지 리스트
            matched_info(Dict[str, int]): stream_ids 당 몇개씩 detection 되었는지 카운트를 알 수 있는 변수

        NOTE:
            categories_batches
                >>> [0, 1, 0, 0]  # 0은 월담, 1은 따라들어가기 순으로 배치 들어온것임. len(batches)보다 클 수 있음
            divided_roi_info[0][2]
                >>> array([[456, 411],
                    [458, 411],
                    [633, 419],
                    [612, 205],
                    [465, 175]], dtype=int32)
            # 0번째 categories_batches의 2번째 ROI 좌표. 이안에 bbox가 검출되었는지 확인하는 용도로 쓰인다.
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
        matched_info = {}  # 스트림별 카운트 (누적)
        cropped_images = []
        total_bbox_info = []

        # [수정 3] Tracking 결과를 기반으로 정보 추출
        # tracker 결과는 Numpy array이므로 .item() 등의 텐서 메서드가 필요 없습니다.
        for cropped_batch_idx, batch_idx_info in enumerate(categories_batches):

            # 현재 처리 중인 Crop이 속한 원본 Stream ID
            current_stream_id = stream_ids[batch_idx_info]

            # matched_info 초기화 (없으면 0)
            if current_stream_id not in matched_info:
                matched_info[current_stream_id] = 0

            now_bbox_info = []

            # 해당 배치의 추적 결과 가져오기 (없으면 빈 배열)
            track_results = tracked_nms_od_result[cropped_batch_idx]

            for trk in track_results:
                # OCSort 출력 포맷: [x1, y1, x2, y2, track_id]
                # (주의: conf, cls 정보는 Tracker 설정에 따라 없을 수 있음. 기본값 기준 ID로 대체)
                x1, y1, x2, y2, track_id = map(int, trk[:5])

                # 좌표 유효성 검사
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue

                # 좌표 원복 (Resize -> Original)
                origin_bbox = modified2origin_coordinate(
                    xyxy=(x1, y1, x2, y2),
                    now_shape=OD_INPUT_SIZE,
                    original_shape=cropped_batches[batch_idx_info].shape[1:3],
                )

                # [중요] 기존 [conf, cls] 대신 [track_id] 정보를 추가합니다.
                # 만약 conf, cls가 꼭 필요하다면 tracker 코드를 수정해야 하지만, 보통은 ID가 더 중요합니다.
                # 필요 시: origin_bbox.extend([track_id, 1.0, 0]) 처럼 더미값이나 매칭 로직 추가 필요
                origin_bbox.append(track_id)

                now_bbox_info.append(origin_bbox)

                # Crop 이미지 추출 (원복된 좌표 사용)
                ox1, oy1, ox2, oy2 = origin_bbox[:4]

                # 좌표가 이미지 범위를 벗어나지 않도록 클리핑 (안전장치)
                h, w = cropped_batches[batch_idx_info].shape[1:3]
                ox1, oy1 = max(0, ox1), max(0, oy1)
                ox2, oy2 = min(w, ox2), min(h, oy2)

                cropped_image = cropped_batches[batch_idx_info][:, oy1:oy2, ox1:ox2]
                cropped_images.append(cropped_image)

                # 카운트 증가
                matched_info[current_stream_id] += 1

            total_bbox_info.append(now_bbox_info)

        return (
            cropped_images,
            matched_info,
            total_bbox_info,
            cropped_batches,
            categories_batches,
            track_info,
        )

    def get_classification_result(self, cropped_image: List[np.array], matched_info) -> List[bool]:
        """np.array type의 이미지를 받아 classification task를 수행하고, 결과를 반환합니다.

        Args:
            cropped_image (List[np.array]): Cropped된 이미지 리스트

        Returns:
            List[bool]: 모든 cropped_image에 대해 분류된 결과가 이상상황이면 True, 아니면 False를 반환받은 리스트
        """
        results = []
        raw_cls_results = []

        if len(cropped_image) == 0:
            return [self.ZERO_TENSOR] * len(matched_info), raw_cls_results

        resized_images = self.cls_letterbox_instance(imgs=cropped_image)
        preprocessed_images = resized_images / 255.0

        enhanced_preprocessed_images = enhance_sky_batch_torch(
            preprocessed_images, sat_thresh=255, sat_gamma=2.0, max_value_boost=120  # 기존 설정값
        )

        raw_cls_results = self.classification_instance(enhanced_preprocessed_images)  # len() = N
        results = multi_category_threshold_check(
            raw_cls_results,
            matched_info,
            TARGET_CATEGORY_INDEX,
            self.CLS_CONFIDENCE_THRESHOLD_TENSOR,
        )

        return results, raw_cls_results

    def get_alarm_patient(
        self, cls_results, track_info: Dict[str, List[int]], stream_ids
    ) -> Dict[str, list]:
        """
        Args:
            cls_results: {'1_pia': [False, True], ...} 또는 [tensor([0])] (감지 없음)
            track_info: {'1_pia': [3, 2], ...}

        Returns:
            alarms: {'1_pia': [True, None]} 형태의 딕셔너리
        """

        # [예외 처리] cls_results가 딕셔너리가 아닌 경우 (예: [tensor([0])])
        # 감지된 객체가 없으므로 update 루프를 돌 수 없습니다.
        detected_streams = set()
        if isinstance(cls_results, dict):

            # 1. 모든 카메라의 데이터를 EventManager에 업데이트
            for stream_id, flags in cls_results.items():

                ids = track_info.get(stream_id, [])

                # Tensor라면 리스트로 변환 (안전장치)
                if hasattr(flags, 'tolist'):
                    flags = flags.tolist()
                elif isinstance(flags, list):
                    # 이미 리스트인 경우
                    pass
                else:
                    # 그 외의 경우 (numpy 등) 처리
                    flags = list(flags)

                if len(ids) != len(flags):
                    continue

                # 상태 갱신
                self.alarm_event_manager.update(results=flags, stream_id=stream_id, track_ids=ids)
                detected_streams.add(stream_id)

        # 화면에 객체가 감지되지 않은 카메라 처리
        for stream_id in stream_ids:
            if stream_id not in detected_streams:
                self.alarm_event_manager.update(results=[], stream_id=stream_id, track_ids=[])

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
