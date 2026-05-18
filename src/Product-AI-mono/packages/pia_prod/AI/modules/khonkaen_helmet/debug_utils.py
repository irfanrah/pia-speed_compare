import cv2
import os
import numpy as np
from pia_prod.AI.modules.khonkaen_helmet.config import (
    IMAGE_SAVE_PATH,
    CLS_CONFIDENCE_THRESHOLD,
    TARGET_CATEGORY_INDEX,
)

from pia.utils.devtools.debug_tools import save_snapshot


def save_snapshot_for_odcls(
    images,
    stream_ids,
    bboxes,
    raw_cls_results,
    category_index,
    classify_matched_info,
    video_mode,
    video_instance,
):
    classify_count = 0
    for image, stream_id, bboxes_each_image in zip(images, stream_ids, bboxes):
        image = np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy())
        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, "helmet")

        matched = classify_matched_info.get(stream_id, 0)
        if matched == 0:
            if video_mode and video_instance is not None:
                video_instance.write_frame(image)
            continue

        cls_results = raw_cls_results[classify_count : classify_count + matched]
        is_event = cls_results[:, category_index] >= CLS_CONFIDENCE_THRESHOLD

        for bbox, cls_result in zip(bboxes_each_image, cls_results):
            cls_conf = cls_result[TARGET_CATEGORY_INDEX]
            if cls_conf >= CLS_CONFIDENCE_THRESHOLD:
                label, color = "Not wear", (0, 0, 255)
            else:
                label, color = "normal", (255, 0, 0)

            x1, y1, x2, y2 = map(int, bbox[:4])
            od_conf = f"{bbox[4]:.3f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{label} {cls_conf:.2f} person: {od_conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        save_snapshot(image, save_dir=save_dir) if is_event.any() else None

        classify_count += matched

        if video_mode and video_instance is not None:
            video_instance.write_frame(image)
