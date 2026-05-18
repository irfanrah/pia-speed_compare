import os
import cv2
from pia_prod.AI.modules.yonsei_tailgate.config import IMAGE_SAVE_PATH
import numpy as np
from pia_prod.AI.modules.yonsei_tailgate.func import xyxy2rhombus_for_topview
from pia.utils.devtools.debug_tools import save_snapshot


def save_snapshot_for_tailgate(image, stream_id, bboxes, rois, category_name=None):
    try:
        if rois is not None:
            for roi in rois:
                if roi is not None and len(roi) > 0:
                    cv2.polylines(image, [roi], isClosed=True, color=(255, 0, 0), thickness=2)

        if len(bboxes) != 0:
            for bbox in bboxes:
                rounded_bbox = list(map(round, bbox))
                cv2.rectangle(
                    image,
                    (rounded_bbox[0], rounded_bbox[1]),
                    (rounded_bbox[2], rounded_bbox[3]),
                    (0, 255, 0),
                    2,
                )

                raw_rho_bbox = xyxy2rhombus_for_topview(
                    rounded_bbox[0], rounded_bbox[1], rounded_bbox[2], rounded_bbox[3]
                )
                rho_bbox = []
                for raw_bbox in raw_rho_bbox:
                    rho_bbox.append(list(map(round, raw_bbox)))

                rho_bbox_np = np.array(rho_bbox, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    image, [rho_bbox_np], isClosed=True, color=(255, 255, 255), thickness=3
                )
        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, category_name)
        save_snapshot(image, save_dir=save_dir)

    except Exception as e:
        print(f"Error while drawing bounding boxes or ROIs on the image.: {e}")
