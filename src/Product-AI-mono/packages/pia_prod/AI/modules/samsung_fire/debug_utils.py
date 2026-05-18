from typing import List, Tuple

import os
import cv2
from pia_prod.AI.modules.samsung_fire.config import IMAGE_SAVE_PATH, CATEGORY_NAME
from pia.utils.devtools.debug_tools import save_snapshot


def save_fire_snapshot(
    image,
    stream_id: str,
    output: str,
    result: str,
    bboxes: List[Tuple[int, int, int, int]],
    classes: List[str],
) -> None:
    if image is None:
        return

    for (x1, y1, x2, y2), cls in zip(bboxes, classes):
        color = (0, 0, 255) if cls == "fire" else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(
        image, f"Output: {output}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )
    cv2.putText(
        image, f"Result: {result}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )
    save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, CATEGORY_NAME)
    save_snapshot(image=image, save_dir=save_dir)
