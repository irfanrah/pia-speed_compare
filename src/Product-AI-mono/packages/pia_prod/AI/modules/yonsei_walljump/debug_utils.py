import cv2
import os
from pia_prod.AI.modules.yonsei_walljump.config import (
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
    category_name=None,
):
    try:
        classify_count = 0
        for idx, (image, stream_id, bboxes_each_image) in enumerate(
            zip(images, stream_ids, bboxes)
        ):
            # no bboxes, skip
            if classify_matched_info[stream_id] == 0:
                continue
            is_event = (
                raw_cls_results[classify_count : classify_count + classify_matched_info[stream_id]][
                    :, category_index
                ]
                >= CLS_CONFIDENCE_THRESHOLD
            )

            # no event, skip
            if not sum(is_event):
                classify_count += classify_matched_info[stream_id]
                continue

            else:
                for i, bbox in enumerate(bboxes_each_image):
                    confidence_score = raw_cls_results[classify_count][TARGET_CATEGORY_INDEX]
                    if confidence_score >= CLS_CONFIDENCE_THRESHOLD:
                        label = category_name
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
                    classify_count += 1
                save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, category_name)
                save_snapshot(image, save_dir=save_dir)
    except Exception as e:
        print(f"Error saving snapshot: {e}")
