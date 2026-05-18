from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.global_config import TASKS_PARAM_NAME
from pia_prod.AI.global_config import (
    ROI_KEY,
    ROI_POLYGON_COORDINATES_KEY,
    USER_PARAM_KEY,
    EVENT_KEY,
    CV_EVENT_KEY,
    RET_EVENT_KEY,
    VQA_EVENT_KEY,
)
from typing import Dict
import os
import torch
import gc


def AddStreamModel2dict(
    data: AddStreamModel, tasks_param_name: list = TASKS_PARAM_NAME, decode_dict: dict = None
) -> dict:
    task_dict = {}
    data = dict(data)
    for task in tasks_param_name:
        temp_dict = {}

        if task not in data.keys():
            continue

        for event in data[task]:  # list of ODModel
            event_key = (
                decode_dict[event.name]
                if (decode_dict) and (event.name in decode_dict)
                else event.name
            )
            event.roi = dict(event.roi)  # ROIModel
            temp_dict[event_key] = dict(event)
        task_dict[task] = temp_dict

    for task in tasks_param_name:
        if task not in data.keys():
            continue
        data[task] = task_dict[task]
    return dict(data)


def get_event_type(user_param):
    event_only = {k: v for k, v in user_param[USER_PARAM_KEY].items() if EVENT_KEY in k}
    for k, v in event_only.items():
        if len(v):
            return k
    return None


def get_queue_name(event_type):
    if event_type == CV_EVENT_KEY:
        return os.getenv("BACKEND_CV_RESULT_RABBITMQ_QUEUE_NAME", "cv_queue_dev")
    elif event_type == RET_EVENT_KEY:
        return os.getenv("BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME", "ret_queue_dev")
    elif event_type == VQA_EVENT_KEY:
        return os.getenv("BACKEND_VQA_RESULT_RABBITMQ_QUEUE_NAME", "vqa_queue_dev")


def get_roi_info(user_params: Dict):
    rois = []
    for user_param in user_params:
        for model in user_param[USER_PARAM_KEY][CV_EVENT_KEY].values():
            if len(model[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]):
                roi = model[ROI_KEY][ROI_POLYGON_COORDINATES_KEY]
                roi = [roi[i : i + 2] for i in range(0, len(roi), 2)]
            else:
                roi = []
            rois.append(roi)
    return rois


def threshold_check(raw_cls_results, matched_info, target_category, cls_conf_thres):
    """
    Classification 결과를 받아 임계값(threshold) 이상인지 확인하는 함수.

    Args:
        raw_cls_results (torch.Tensor): Classification 모델의 출력 결과 (N, C) 크기.
                                         (N: 샘플 개수, C: 클래스 개수)
        matched_info (Dict[str, int]): 각 stream_id 별 탐지된 객체 수를 저장한 딕셔너리.
        target_category (int): 검사할 타겟 카테고리 인덱스.
        threshold (float): 이상상황으로 판단할 임계값.

    Returns:
        torch.Tensor: 이벤트 발생 여부를 나타내는 Boolean 텐서.
                      (각 카메라별 1개의 Boolean 값 포함)
    """

    if isinstance(cls_conf_thres, (float, list)):
        cls_conf_thres = torch.tensor(
            cls_conf_thres, dtype=raw_cls_results.dtype, device=raw_cls_results.device
        )

    first_column = raw_cls_results[:, target_category]  # shape: (N,)
    abnormal_flag = first_column >= cls_conf_thres  # shape: (N,), dtype: torch.bool

    results = []  # 임시 보관용 (각 카메라 segment 결과를 담을 0D 텐서 리스트)

    i = 0
    for camera_id, cnt in matched_info.items():
        segment_any = torch.any(abnormal_flag[i : i + cnt])
        results.append(segment_any.unsqueeze(0))
        i += cnt

    result_tensor = torch.cat(results, dim=0)  # shape: [num_cameras], dtype: bool

    return result_tensor


def multi_category_threshold_check(raw_cls_results, matched_info, target_category, cls_conf_thres):
    """
    Classification 결과를 받아 임계값 이상인지 확인하고,
    matched_info를 이용해 카메라 별로 결과를 나누어 반환합니다.

    Args:
        raw_cls_results (torch.Tensor): (N, C) 크기. (N: 전체 객체 수, C: 클래스 개수)
        matched_info (Dict[str, int]): {stream_id: count} 형태. (예: {'1_pia': 3, '2_pia': 2})
        target_category (int or tuple): 검사할 타겟 카테고리 인덱스.
        cls_conf_thres (float): 임계값.

    Returns:
        Dict[str, torch.Tensor]:
            Key: stream_id
            Value: 해당 카메라에서 검출된 객체들의 Boolean Tensor (크기: [count])
    """

    # 1. target_category가 int면 튜플로 변환
    if isinstance(target_category, int):
        target_category = (target_category,)

    # 2. 원하는 타겟 카테고리 컬럼들만 선택 [N, K]
    selected_logits = raw_cls_results[:, target_category]

    # 3. 임계값 비교 (각 클래스별 True/False) [N, K]
    is_exceed_thresh = selected_logits >= cls_conf_thres

    # 4. 객체별로 "하나라도" True인지 확인 (Dim=1 축소) -> [N]
    # 전체 배치에 대한 Flat한 결과입니다.
    flat_abnormal_flags = torch.any(is_exceed_thresh, dim=1)

    # 5. matched_info를 이용하여 카메라별로 결과 쪼개기 (Splitting)
    final_results = {}
    current_idx = 0

    for stream_id, count in matched_info.items():
        # 현재 카메라에 해당하는 부분만 슬라이싱
        # 예: 1_pia가 3개라면 [0:3], 그 다음 2_pia가 2개라면 [3:5]
        camera_flags = flat_abnormal_flags[current_idx : current_idx + count]

        final_results[stream_id] = camera_flags

        # 인덱스 이동
        current_idx += count

    return final_results


def free_autobackend(ab):
    if ab is None:
        return
    # 0) 안전하게 동기화
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    # 1) 백엔드별 내부 핸들 정리 (존재하는 것만)
    # - Torch: model
    try:
        if hasattr(ab, "model") and ab.model is not None:
            try:
                ab.model.to("cpu")
            except Exception:
                pass
            ab.model = None
    except Exception:
        pass

    # - TensorRT: engine/context/stream/bindings 등
    for name in (
        "context",
        "engine",
        "stream",
        "bindings",
        "binding_addrs",
        "allocator",
        "runtime",
        "session",
    ):
        if hasattr(ab, name):
            try:
                setattr(ab, name, None)
            except Exception:
                pass

    # - 기타 close()가 있으면 호출
    if hasattr(ab, "close") and callable(ab.close):
        try:
            ab.close()
        except Exception:
            pass

    # 2) 남은 참조 끊기
    del ab
    gc.collect()

    # 3) CUDA 캐시 반환
    try:
        torch.cuda.empty_cache()
        # (Optional) 프로세스 간 공유 메모리 반환
        torch.cuda.ipc_collect()
    except Exception:
        pass

    # 4) 마지막 동기화 (선택)
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
