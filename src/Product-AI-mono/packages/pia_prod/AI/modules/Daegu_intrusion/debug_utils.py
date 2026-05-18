import cv2
import os
import numpy as np
from pia_prod.AI.modules.Daegu_intrusion.config import (
    IMAGE_SAVE_PATH,
)

from pia.utils.devtools.debug_tools import save_snapshot


def save_snapshot_for_od(
    images,
    stream_ids,
    bboxes,
    category_index,
    video_mode,
    video_instance,
):
    for image, stream_id, bboxes_each_image in zip(images, stream_ids, bboxes):
        is_something = False
        image = np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy())
        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, "intrusion")
        matched = len(bboxes_each_image)
        if matched == 0:
            if video_mode and video_instance is not None:
                video_instance.write_frame(image)
            continue
        is_something = True

        for bbox in bboxes_each_image:
            x1, y1, x2, y2 = map(int, bbox[:4])
            color = (0, 0, 255)
            od_conf = f"{bbox[4]:.3f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"Person: {od_conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        save_snapshot(image, save_dir=save_dir) if is_something else None

        if video_mode and video_instance is not None:
            video_instance.write_frame(image)
