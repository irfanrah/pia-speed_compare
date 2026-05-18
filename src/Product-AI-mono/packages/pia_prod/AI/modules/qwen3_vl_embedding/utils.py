import cv2
import torch
import numpy as np
from typing import Optional

def draw_status(frame, anomaly_text: Optional[str], frame_idx: int, width: int, height: int):
    """
    Draw:
    - Status text at TOP-LEFT
    - Frame index at BOTTOM-LEFT
    """

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ===============================
    # 1) TOP-LEFT : status text
    # ===============================
    status_x, status_y = 10, 30
    status_scale = 0.9
    status_thick = 2

    if anomaly_text is None:
        status_text = "PRED: -"
        status_color = (160, 160, 160)   # gray
    else:
        status_text = f"PRED: {anomaly_text}"
        if "normal" in anomaly_text.lower():
            status_color = (0, 255, 0)   # green
        else:
            status_color = (0, 0, 255)   # red

    cv2.putText(
        frame,
        status_text,
        (status_x, status_y),
        font,
        status_scale,
        status_color,
        status_thick,
        cv2.LINE_AA,
    )

    # ===============================
    # 2) BOTTOM-LEFT : frame index
    # ===============================
    idx_text = f"frame={frame_idx}"
    idx_scale = 0.7
    idx_thick = 2

    # baseline-aware placement
    (tw, th), baseline = cv2.getTextSize(idx_text, font, idx_scale, idx_thick)
    idx_x = 10
    idx_y = height - 10  # bottom margin

    cv2.putText(
        frame,
        idx_text,
        (idx_x, idx_y),
        font,
        idx_scale,
        (255, 255, 255),
        idx_thick,
        cv2.LINE_AA,
    )