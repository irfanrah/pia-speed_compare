import cv2


ALARM_ON_COLOR = (0, 0, 255)
ALARM_OFF_COLOR = (0, 255, 0)
OUTLINE_COLOR = (0, 0, 0)


def _outlined_text(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(
        frame, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, OUTLINE_COLOR, thick + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA,
    )


def draw_multi_category_status(
    frame, frame_idx, width, height,
    category_rows, queue_size, threshold,
):
    """
    Render one row per category, stacked from top-left downward.

    Works for single-category (one-entry list) and multi-category (many-entry list).
    Each category's label is unique so colors are fixed — green when alarm is off,
    red when alarm is on, both with a black outline for contrast.

    - category_rows: list of (label, queue_list, triggered)
    - queue_list: iterable of 0/1 (current alarm queue contents for this category)
    - triggered: bool (sum(queue_list) >= threshold)

    Each row layout:
        [LABEL]  (ON|OFF badge)  [queue-cells with threshold tick]  q=[..] thr=T/Q
    Frame index is drawn at bottom-left.
    """
    x0 = 10
    row_y = 30
    row_height = 46
    label_w = 140
    badge_w = 64
    bar_w = 200
    bar_h = 18

    for label, cq, triggered in category_rows:
        color = ALARM_ON_COLOR if triggered else ALARM_OFF_COLOR
        badge_text = "ON" if triggered else "OFF"
        cq_list = list(cq) if cq else []

        _outlined_text(frame, f"[{label.upper()}]", (x0, row_y), 0.7, color, 2)

        badge_x = x0 + label_w
        badge_tl = (badge_x, row_y - 20)
        badge_br = (badge_x + badge_w, row_y + 6)
        cv2.rectangle(frame, badge_tl, badge_br, color, -1)
        cv2.rectangle(frame, badge_tl, badge_br, OUTLINE_COLOR, 2)
        _outlined_text(frame, badge_text, (badge_x + 8, row_y), 0.65, (255, 255, 255), 2)

        bar_x = badge_x + badge_w + 15
        bar_y = row_y - 16
        cell_w = bar_w // max(queue_size, 1)
        for i in range(queue_size):
            cx1 = bar_x + i * cell_w
            cx2 = cx1 + cell_w - 2
            cy1 = bar_y
            cy2 = bar_y + bar_h
            if i < len(cq_list):
                fill = ALARM_ON_COLOR if cq_list[i] else ALARM_OFF_COLOR
            else:
                fill = (80, 80, 80)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), fill, -1)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), OUTLINE_COLOR, 1)

        thr_x = bar_x + threshold * cell_w - 1
        cv2.line(
            frame,
            (thr_x, bar_y - 3),
            (thr_x, bar_y + bar_h + 3),
            (0, 255, 255), 2,
        )

        q_text = f"q={cq_list} thr={threshold}/{queue_size}"
        _outlined_text(
            frame, q_text,
            (bar_x + bar_w + 15, row_y),
            0.55, color, 2,
        )
        row_y += row_height

    _outlined_text(
        frame, f"frame={frame_idx}",
        (10, height - 10),
        0.7, (255, 255, 255), 2,
    )
