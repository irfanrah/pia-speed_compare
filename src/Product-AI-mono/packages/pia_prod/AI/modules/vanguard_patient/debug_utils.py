import cv2
import os
import torch
import numpy as np
from pia_prod.AI.modules.vanguard_patient.config import (
    IMAGE_SAVE_PATH,
    CLS_CONFIDENCE_THRESHOLD,
    CLASSIFY_DICT,
    OD_INPUT_SIZE,
)
from pia.utils.devtools.debug_tools import save_snapshot


def save_snapshot_for_odtrackercls(
    images,
    stream_ids,
    total_bbox_info,
    raw_cls_results,
    matched_info,
    category_index,
    roi_dict=None,
    event_result=None,  # [추가] 이벤트 상태 딕셔너리 {'stream_id': int}
    video_mode=False,
    video_instance=None,
):
    """
    Object Detection + Tracking + Classification + ROI + Alarm Status 시각화
    """

    global_cls_idx = 0

    if isinstance(category_index, int):
        target_indices = (category_index,)
    else:
        target_indices = category_index

    for image, stream_id, bboxes in zip(images, stream_ids, total_bbox_info):

        # 1. Image Conversion
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().permute(1, 2, 0).numpy()

        image = np.ascontiguousarray(image).copy()
        img_h, img_w = image.shape[:2]

        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, "patient_monitor")

        # -----------------------------------------------------------
        # [1] ROI 그리기 (녹색)
        # -----------------------------------------------------------
        if roi_dict is not None and stream_id in roi_dict:
            roi_poly_lb = roi_dict[stream_id].get("after_letterbox_calc_origin_roi")

            if roi_poly_lb is not None and len(roi_poly_lb) > 0:
                roi_poly_restored = undo_letterbox_roi(
                    roi_poly_lb, src_shape=(img_h, img_w), target_shape=OD_INPUT_SIZE
                )
                pts = roi_poly_restored.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # -----------------------------------------------------------
        # [2] 알람 상태 시각화 (빨간 테두리 + 텍스트)
        # -----------------------------------------------------------
        if event_result is not None:
            # 딕셔너리에서 현재 스트림의 상태 가져오기 (기본값 0)
            status = event_result.get(stream_id, 0)

            # 상태가 1(발생) 또는 2(지속)일 때만 그리기
            if status in [1, 2]:
                # 1. 화면 테두리 그리기 (두께 3, 빨강)
                # cv2.rectangle은 두께가 안쪽으로 그려지지 않고 중심 기준이라
                # 이미지를 벗어나지 않게 약간 안쪽 좌표를 잡음
                border_thickness = 3
                cv2.rectangle(
                    image,
                    (0, 0),
                    (img_w - 1, img_h - 1),
                    (0, 0, 255),  # Red
                    thickness=border_thickness * 4,  # 넉넉하게 그려야 잘 보임
                )

                # 2. 알람 텍스트 "ALARM: PATIENT"
                alarm_text = "ALARM: PATIENT"
                font_scale = 1.0
                font_thickness = 2
                text_color = (0, 0, 255)  # Red

                # 텍스트 배경 (가독성)
                (t_w, t_h), _ = cv2.getTextSize(
                    alarm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                cv2.rectangle(image, (0, 0), (t_w + 10, t_h + 20), (255, 255, 255), -1)  # 흰색 배경

                cv2.putText(
                    image,
                    alarm_text,
                    (5, t_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                )

        # -----------------------------------------------------------
        # [3] 객체 및 Classification 결과 그리기
        # -----------------------------------------------------------
        num_objects = matched_info.get(stream_id, 0)

        if num_objects == 0:
            if video_mode and video_instance is not None:
                video_instance.write_frame(image)
            save_snapshot(image, save_dir=save_dir)
            continue

        current_cls_results = raw_cls_results[global_cls_idx : global_cls_idx + num_objects]
        global_cls_idx += num_objects

        selected_probs = current_cls_results[:, target_indices]
        is_event_per_object = (selected_probs >= CLS_CONFIDENCE_THRESHOLD).any(dim=1)

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox[:4])
            track_id = int(bbox[-1])

            is_abnormal = is_event_per_object[i].item()

            probs = current_cls_results[i]
            max_class_idx = torch.argmax(probs).item()
            max_conf = probs[max_class_idx].item()
            predicted_class_name = CLASSIFY_DICT.get(max_class_idx, "Unknown")

            if is_abnormal:
                color = (0, 0, 255)  # Red
                label_prefix = "[!] "
            else:
                color = (255, 0, 0)  # Blue
                label_prefix = ""

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            text_caption = f"ID:{track_id} {label_prefix}{predicted_class_name} ({max_conf:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(text_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(image, (x1, y1 - 25), (x1 + text_w, y1), color, -1)

            cv2.putText(
                image,
                text_caption,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # 5. 저장
        save_snapshot(image, save_dir=save_dir)

        if video_mode and video_instance is not None:
            video_instance.write_frame(image)


def undo_letterbox_roi(roi_poly, src_shape, target_shape=(640, 640)):
    """
    Letterbox(Padding + Resize)된 ROI 좌표를 원본 이미지 좌표로 복구합니다.

    Args:
        roi_poly (np.ndarray): (N, 2) 형태의 Letterbox 기준 좌표
        src_shape (tuple): 원본 이미지 크기 (Height, Width) -> cropped_image.shape[:2]
        target_shape (tuple): 모델 입력 크기 (Height, Width) -> OD_INPUT_SIZE (예: 640, 640)

    Returns:
        np.ndarray: 복구된 좌표 (N, 2)
    """
    if roi_poly is None or len(roi_poly) == 0:
        return np.array([])

    h, w = src_shape
    t_h, t_w = target_shape

    # 1. Scale 비율 계산 (Letterbox 로직과 동일)
    r = min(t_h / h, t_w / w)

    # 2. Padding 크기 계산
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    dw = (t_w - new_unpad_w) / 2  # 가로 패딩 (좌우 합쳐서)
    dh = (t_h - new_unpad_h) / 2  # 세로 패딩 (상하 합쳐서)

    # 3. 역변환 (Coordinate Mapping)
    # x_original = (x_letterbox - dw) / r
    # y_original = (y_letterbox - dh) / r

    restored_roi = np.zeros_like(roi_poly, dtype=np.float32)
    restored_roi[:, 0] = (roi_poly[:, 0] - dw) / r
    restored_roi[:, 1] = (roi_poly[:, 1] - dh) / r

    return restored_roi.astype(np.int32)
