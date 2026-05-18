import cv2
import os
from pia_prod.AI.modules.yeonsei_smoke.config import (
    IMAGE_SAVE_PATH,
    SMOKE_THRESHOLD,
)
from pia.utils.devtools.debug_tools import save_snapshot


def save_smoke_snapshot(image, stream_id, category_name, result, bboxes):
    if image is not None:
        for i, bbox in enumerate(bboxes):
            confidence_score = result[i]
            if confidence_score > SMOKE_THRESHOLD:
                label = "smoke"
                color = (0, 0, 255)
            else:
                label = "normal"
                color = (255, 0, 0)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                image,
                f"{label} {confidence_score:.2f}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, category_name)

        save_snapshot(image=image, save_dir=save_dir)
