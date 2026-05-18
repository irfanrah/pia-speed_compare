import cv2
from typing import Optional

def draw_status(
    frame,
    anomaly_text: Optional[str],
    frame_idx: int,
    width: int,
    height: int,
    alarm_queue_count: Optional[int] = None,
    alarm_queue_size: Optional[int] = None,
    alarm_threshold: Optional[int] = None,
    is_triggered: Optional[bool] = None,
    alarm_queue_entries: Optional[list] = None,
):
    """
    Draw:
    - Status text at TOP-LEFT
    - Alarm queue bar + trigger state below status
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
    # 2) ALARM QUEUE visualization
    # ===============================
    if alarm_queue_size is not None and alarm_threshold is not None:
        q_count = alarm_queue_count if alarm_queue_count is not None else 0
        info_y = 65
        info_scale = 0.7
        info_thick = 2

        # Queue count text: "Queue: 2/3 (thr: 2)"
        q_fill = len(alarm_queue_entries) if alarm_queue_entries is not None else q_count
        queue_text = f"Queue: {q_count}/{alarm_queue_size} (thr: {alarm_threshold})"
        cv2.putText(
            frame, queue_text, (10, info_y),
            font, info_scale, (255, 255, 255), info_thick, cv2.LINE_AA,
        )

        # Draw queue bar — each cell reflects actual queue entry
        bar_x, bar_y = 10, info_y + 10
        bar_w, bar_h = 200, 18
        cell_w = bar_w // alarm_queue_size

        for i in range(alarm_queue_size):
            x1 = bar_x + i * cell_w
            x2 = x1 + cell_w - 2
            y1 = bar_y
            y2 = bar_y + bar_h
            if alarm_queue_entries is not None and i < len(alarm_queue_entries):
                if alarm_queue_entries[i]:
                    color = (0, 0, 255)   # red = anomalous
                else:
                    color = (0, 180, 0)   # green = normal (filled)
            elif i < q_count:
                color = (0, 0, 255)       # red = anomalous (fallback)
            else:
                color = (80, 80, 80)      # dark gray = empty
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Threshold line on the bar
        thr_x = bar_x + alarm_threshold * cell_w - 1
        cv2.line(frame, (thr_x, bar_y - 3), (thr_x, bar_y + bar_h + 3), (0, 255, 255), 2)

        # Trigger state — show ON/OFF with queue count
        trigger_y = bar_y + bar_h + 25
        if is_triggered is True:
            trigger_text = f"STATE: ON ({q_count}/{alarm_queue_size})"
            trigger_color = (0, 0, 255)    # red
        elif is_triggered is False:
            trigger_text = f"STATE: OFF ({q_count}/{alarm_queue_size})"
            trigger_color = (0, 255, 0)    # green
        else:
            trigger_text = "STATE: -"
            trigger_color = (160, 160, 160)

        cv2.putText(
            frame, trigger_text, (10, trigger_y),
            font, info_scale, trigger_color, info_thick, cv2.LINE_AA,
        )

    # ===============================
    # 3) BOTTOM-LEFT : frame index
    # ===============================
    idx_text = f"frame={frame_idx}"
    idx_scale = 0.7
    idx_thick = 2

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
