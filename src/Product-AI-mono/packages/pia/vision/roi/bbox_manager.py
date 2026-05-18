from typing import List

import numpy as np
from pia.ai.tasks.tracker.models.sort.sort import distance_batch, iou_batch
from ultralytics.utils.plotting import Annotator, colors


class BoundingBoxManager:
    def __init__(self, iou_threshold=0.1, distance_threshold=100):
        """
        The distance threshold is applied depending on resolution of the original image
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.index = 0

    def merge_boxes(self, boxes: np.array) -> np.array:
        """
        input boxes -> [[x1, y1, x2, y2, conf, category], [x1, y1, x2, y2, conf, category], [x1, y1, x2, y2, conf, category], ....]
        output box  -> [xmin, ymin, xmax, ymax, conf, category]
        TODO: conf and category are not used -> How to use?
        """
        x1 = min(boxes[:, 0])
        y1 = min(boxes[:, 1])
        x2 = max(boxes[:, 2])
        y2 = max(boxes[:, 3])
        confidence_score = boxes[0, -2]  # Not used, just temporarily
        cat = boxes[0, -1]  # Not used, just temporarily
        return np.array([x1, y1, x2, y2, confidence_score, cat], dtype=np.float32)

    def do_merge(self, boxes, merge_func, merge_threshold):
        """
        clac cost between boxes
        has two costs
            1. EUC(EUClidean distance) between bboxes - The large the value , the Farther
            2. IOU(Intersection Over Union) between bboxes - The smaller the value, the farther
        has two thresholds
            1. EUC - between 0 ~ inf (TODO : need to set the max value )
            2. IOU - betwwen 0 ~ 1
        """
        merged_boxes = []
        distances = merge_func(boxes, boxes)
        copy_boxes = boxes.copy()
        indices_to_delete = set()

        for i in range(len(distances)):
            if i in indices_to_delete:
                continue
            if merge_func == distance_batch:
                condition = np.argwhere(distances[i] <= merge_threshold).flatten()
            elif merge_func == iou_batch:
                condition = np.argwhere(distances[i] >= merge_threshold).flatten()
            condition = condition[condition != i]  # Not included self

            if len(condition):
                indices_to_delete.update(condition)
                indices_to_delete.add(i)
                # Merge, including the boxes to be merged
                merge_candidates = np.vstack([copy_boxes[i], copy_boxes[condition]])
                merged_box = self.merge_boxes(merge_candidates)
                merged_boxes.append(merged_box)

        remaining_boxes = np.delete(copy_boxes, list(indices_to_delete), axis=0)

        if len(remaining_boxes) > 0:
            merged_boxes.extend(remaining_boxes)

        return (
            np.array(merged_boxes) if len(merged_boxes) > 0 else np.empty((0, 4), dtype=np.float32)
        )

    def update_boxes(self, boxes: np.array):
        # IOU based merge
        if (
            self.iou_threshold < 1.0 and self.iou_threshold > 0
        ):  # if the threshold exceeds 1 or negative, not used
            boxes = self.do_merge(boxes, iou_batch, self.iou_threshold)

        # distance based merge
        if self.distance_threshold > 0:  # if the threshold smaller then 0, not used
            boxes = self.do_merge(boxes, distance_batch, self.distance_threshold)
        return boxes

    @staticmethod
    def draw_bbox(
        img: np.ndarray, boxes: List[List[int]], clss: List[int], names: List[str]
    ) -> np.ndarray:
        """Draw bounding boxes on the image

        Args:
            img (np.ndarray): Input image in format (H, W, C)
            boxes (List[List[int]]): List of boxes in format [x1, y1, x2, y2]
            clss (List[int]): List of class indices for each box
            names (List[str]): List of names for each class

        Returns:
            np.ndarray: Image with bounding boxes

        Example:
            >>> img = np.random.randint(0, 255, size=(224, 224, 3))
            >>> boxes = [[10, 10, 80, 80], [90, 90, 200, 200]]
            >>> clss = [0, 1]
            >>> names = ["person", "car"]
            >>> annotated_img = draw_bbox(img, boxes, clss, names)
        """
        img = img.astype(np.uint8)
        annotator = Annotator(img, line_width=2)

        for box, cls in zip(boxes, clss):
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

        return annotator.im
