import cv2
from PIL import Image, ImageDraw

import cv2
import os
import numpy as np
from pia_prod.AI.modules.vehicle_reverse.config import IMAGE_SAVE_PATH

from pia.utils.devtools.debug_tools import save_snapshot


def save_snapshot_for_vehicle_reverse(
        images,
        stream_ids,
        events_list,
        video_mode,
        video_instance,
    ):
    for image, stream_id, events in zip(images, stream_ids, events_list):
        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, "vehicle_reverse")

        if not events:
            if video_mode and video_instance is not None:
                video_instance.write_frame(image)
            continue

        for event in events:
            x1, y1, w, h = map(int, event["bbox"])
            x2, y2 = x1 + w, y1 + h
            color = (0, 0, 255) if event["is_wrong"] else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"ID:{event['id']}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        save_snapshot(image, save_dir=save_dir)

        if video_mode and video_instance is not None:
            video_instance.write_frame(image)
