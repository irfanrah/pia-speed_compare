from typing import Union

import cv2
import numpy as np
import torch


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default)
    from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple):
            The shape of the image that the bounding boxes are for,
            in the format of (height, width).
        boxes (torch.Tensor):
            the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple):
            the shape of the target image, in the format of (height, width).
        ratio_pad (tuple):
            a tuple of (ratio, pad) for scaling the boxes. If not provided,
            the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool):
            If True, assuming the boxes is based on image augmented by yolo style.
            If False then do regular rescaling.
        xywh (bool):
            The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def calc_expand_coord(
    roi,
    frame_wh,
    expand_ratio: float = 0.0,
    horizontal_px: Union[int, list] = 0,
    vertical_px: Union[int, list] = 0,
):
    if len(roi) == 0:
        return []

    if not isinstance(horizontal_px, list):
        horizontal_px = [horizontal_px, horizontal_px]
    if not isinstance(vertical_px, list):
        vertical_px = [vertical_px, vertical_px]

    xmax = np.array(roi)[:, 0].max(axis=0)
    ymax = np.array(roi)[:, 1].max(axis=0)
    xmin = np.array(roi)[:, 0].min(axis=0)
    ymin = np.array(roi)[:, 1].min(axis=0)
    width = xmax - xmin
    height = ymax - ymin

    re_xmin = max(0, int(xmin - (width * expand_ratio + horizontal_px[0])))
    re_ymin = max(0, int(ymin - (height * expand_ratio + vertical_px[0])))
    re_xmax = min(frame_wh[0], int(xmax + (width * expand_ratio + horizontal_px[1])))
    re_ymax = min(frame_wh[1], int(ymax + (height * expand_ratio + vertical_px[1])))

    return np.array(
        [[re_xmin, re_ymin], [re_xmin, re_ymax], [re_xmax, re_ymax], [re_xmax, re_ymin]]
    )


def preprocess(im, device, half=False):
    """Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = im[None, :]
        im = im.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    img = im.to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    if not_tensor:
        img /= 255  # 0 - 255 to 0.0 - 1.0
    return img


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(
        self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
    ):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, image=None, labels=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        self.origin_shape = shape
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        self.ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        self.dw, self.dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
        if self.auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, self.stride), np.mod(
                self.dh, self.stride
            )  # wh padding
        elif self.scaleFill:  # stretch
            self.dw, self.dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            self.ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (
                labels["ratio_pad"],
                (self.dw, self.dh),
            )  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, self.ratio, self.dw, self.dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

    def get_origin_size_bbox(self, resized_bboxes: np.ndarray):
        resized_bboxes[:, [0, 2]] = (resized_bboxes[:, [0, 2]] - self.dw) * (
            1 / self.ratio[0]
        )
        resized_bboxes[:, [1, 3]] = (resized_bboxes[:, [1, 3]] - self.dh) * (
            1 / self.ratio[1]
        )
        return resized_bboxes
