import cv2
import os
import torch
import numpy as np
from pia.utils.devtools.debug_tools import save_snapshot
from pia_prod.AI.modules.aramco_loitering.config import OD_INPUT_SIZE, IMAGE_SAVE_PATH


def save_snapshot_for_odtracker(
    images,
    stream_ids,
    total_bbox_info,
    event_manager,
    roi_dict=None,
    video_mode=False,
    video_instance=None,
):
    """
    Object Detection + Tracking + Loitering Event 시각화 (logging_flag True 일 때만 실행됨)
    """

    for image, stream_id, bboxes in zip(images, stream_ids, total_bbox_info):

        # 1. Image Conversion (Tensor -> Numpy)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().permute(1, 2, 0).numpy()

        image = np.ascontiguousarray(image).copy()
        img_h, img_w = image.shape[:2]

        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, "loitering_monitor")

        # -----------------------------------------------------------
        # [1] ROI 그리기 (녹색 폴리곤)
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
        # [2] 알람 상태 시각화 (빨간 테두리 + 경고 텍스트)
        # -----------------------------------------------------------
        # event_status 상태: 이벤트 발생(1) 이나 지속(3)일 때 알람 표시
        status = event_manager.event_status.get(stream_id, 0)

        if status in [1, 3]:
            border_thickness = 3
            cv2.rectangle(
                image,
                (0, 0),
                (img_w - 1, img_h - 1),
                (0, 0, 255),  # Red
                thickness=border_thickness * 4,
            )

            alarm_text = "ALARM: LOITERING DETECTED"
            font_scale = 1.0
            font_thickness = 2

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
                (0, 0, 255),
                font_thickness,
            )

        # -----------------------------------------------------------
        # [3] 객체 Bounding Box 및 게이지 (count / max_score) 텍스트 그리기
        # -----------------------------------------------------------
        max_score = event_manager.max_score
        threshold = event_manager.threshold

        # 해당 카메라에 저장된 큐 정보(점수)들 가져오기
        stream_durations = event_manager.duration_queue.get(stream_id, {})

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            track_id = int(bbox[-1])

            # Event Manager에서 현재 이 사람의 체류 점수 가져오기 (없으면 0)
            current_score = stream_durations.get(track_id, 0)

            # 임계치를 넘었는지 여부에 따라 색상 결정 (배회 확정이면 빨강, 아니면 파랑)
            is_loitering = current_score >= threshold
            color = (0, 0, 255) if is_loitering else (255, 0, 0)

            # 1. Bounding Box 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 2. 머리 위에 (count / max_score) 텍스트 그리기
            text_caption = f"ID:{track_id} ({int(current_score)}/{int(max_score)})"

            (text_w, text_h), _ = cv2.getTextSize(text_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # 텍스트 가독성을 위해 뒷배경 채우기
            cv2.rectangle(image, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
            cv2.putText(
                image,
                text_caption,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # 흰색 글씨
                2,
            )

        # -----------------------------------------------------------
        # [4] 저장 및 비디오 출력
        # -----------------------------------------------------------
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
