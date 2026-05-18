from typing import List

import numpy as np
import torch
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import scale_boxes
from pia.vision.postprocessing.nms import non_max_suppression


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple):
            A tuple of integers representing the size of the image
            in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
    """
    if isinstance(
        coords, torch.Tensor
    ):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple):
            The shape of the image that the coords are from.
        coords (torch.Tensor):
            the coords to be scaled of shape n,2.
        img0_shape (tuple):
            the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple):
            the ratio of the image size to the padded image size.
        normalize (bool):
            If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool):
            If True, assuming the boxes is based on image augmented by yolo style.
            If False then do regular rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def get_keypoint_result(preds: List[torch.Tensor], orig_imgs: List[np.ndarray], resize_size):
    preds = non_max_suppression(
        preds,
        0.7,
        0.25,
        agnostic=False,
        max_det=300,
        classes=None,
        nc=1,
    )
    results = []
    for pred, orig_img in zip(preds, orig_imgs):
        if len(pred):
            pred[:, :4] = scale_boxes(resize_size, pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *(17, 3)) if len(pred) else pred[:, 6:]
            pred_kpts = scale_coords(resize_size, pred_kpts, orig_img.shape)[:, :, :2].round()
            boxes = pred[:, :6]
            results.append([pred_kpts.detach().cpu().numpy(), boxes.detach().cpu().numpy()])
        else:
            results.append([])
    return results
