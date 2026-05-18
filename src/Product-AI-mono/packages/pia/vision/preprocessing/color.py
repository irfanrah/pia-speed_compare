from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import cv2
import numpy as np
import torch
from numba import njit, prange

code_map = {
    # "bgr": None, # 기본값을 bgr로 가정하여 변환하지 않음
    "rgb": cv2.COLOR_BGR2RGB,
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
    "ycrcb": cv2.COLOR_BGR2YCrCb,
    "gray": cv2.COLOR_BGR2GRAY,
    "luv": cv2.COLOR_BGR2LUV,
    "xyz": cv2.COLOR_BGR2XYZ,
    "hls": cv2.COLOR_BGR2HLS,
    "yuv": cv2.COLOR_BGR2YUV,
}


@njit(parallel=True, fastmath=True)
def _batch_and_numba(out, a, b):
    for n in prange(a.shape[0]):
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                out[n, i, j] = a[n, i, j] & b[n, i, j]


def numba_batch_and(batch1, batch2):
    a = np.stack(batch1)
    b = np.stack(batch2)
    out = np.empty_like(a)
    _batch_and_numba(out, a, b)
    return [out[i] for i in range(out.shape[0])]


@njit(parallel=True, fastmath=True)
def _batch_or_numba(out, a, b):
    for n in prange(a.shape[0]):
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                out[n, i, j] = a[n, i, j] | b[n, i, j]


def numba_batch_or(batch1, batch2):
    a = np.stack(batch1)
    b = np.stack(batch2)
    out = np.empty_like(a)
    _batch_or_numba(out, a, b)
    return [out[i] for i in range(out.shape[0])]


# (B, H, W, 3) ── channel-last
@njit(parallel=True, fastmath=True, cache=True)
def numba_bgr2rgb_batch_cl(images: np.ndarray):
    out = np.empty_like(images)
    B = images.shape[0]
    for i in prange(B):
        out[i, :, :, 0] = images[i, :, :, 2]  # R ← B
        out[i, :, :, 1] = images[i, :, :, 1]  # G
        out[i, :, :, 2] = images[i, :, :, 0]  # B ← R
    return out


# (B, 3, H, W) ── channel-first
@njit(parallel=True, fastmath=True, cache=True)
def numba_bgr2rgb_batch_cf(images: np.ndarray):
    out = np.empty_like(images)
    B = images.shape[0]
    for i in prange(B):
        out[i, 0, :, :] = images[i, 2, :, :]  # R ← B
        out[i, 1, :, :] = images[i, 1, :, :]  # G
        out[i, 2, :, :] = images[i, 0, :, :]  # B ← R
    return out


def cv_bgr2rgb_batch(images: list):
    for im in images:
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB, im)  # in-place


def torch_bgr2rgb_batch_cf(tensor: torch.Tensor):  # channel-first (B,3,H,W)
    return tensor[:, [2, 1, 0], :, :]  # view, zero-copy


def torch_bgr2rgb_batch_cl(tensor: torch.Tensor):  # channel-last (B,H,W,3)
    return tensor[:, :, :, [2, 1, 0]]  # view, zero-copy


# =========================
# convert_colors
# =========================
def convert_colors(
    img: np.ndarray,
    conditions: list[str],
) -> DefaultDict[str, np.ndarray]:
    """
    다양한 색상 공간 변환을 조건적으로 반환.
    conditions 키: {'rgb','hsv','lab','ycrcb','gray','luv','xyz','hls','yuv'}
    예)
        conditions = {'rgb': True, 'hsv': False, 'lab': True}
    반환:
        defaultdict(np.ndarray) 형태로 요청된 키마다 변환 이미지를 append.
    """
    result: Dict[str, np.ndarray] = {}
    for space in conditions:
        if space not in code_map:
            continue
        code = code_map[space]
        converted = cv2.cvtColor(img, code)
        result[space] = converted
    return result


def batch_convert_colors(
    batch: List[np.ndarray],
    conditions: List[str]
) -> DefaultDict[str, List[np.ndarray]] :
    result: dict[str, List[np.ndarray]] = defaultdict(list)
    for img in batch:
        ret = convert_colors(
            img=img,
            conditions=conditions
        )
        for condition, cvt_im in ret.items():
            result[condition].append(cvt_im)
    return result


def color_filtering(
    frame: np.ndarray,
    lower: Tuple[np.ndarray],
    upper: Tuple[np.ndarray],
    kernel_size: Tuple[int, int],
    operations: List[Tuple[str, int]] = None,
) -> np.ndarray:
    """
    단일 프레임 색상 마스크.
    - lower/upper: 색상 범위
    - kernel_size: 커널 크기
    - operations: 형태학 연산 리스트 (순서대로 적용)
        예) [("close", 2), ("open", 1), ("dilate", 1)]
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.inRange(frame, lower, upper)

    if not operations:
        return mask

    for op, iters in operations:
        if iters <= 0:
            continue
        if op == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
        elif op == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        elif op == "dilate":
            mask = cv2.dilate(mask, kernel, iterations=iters)
        elif op == "erode":
            mask = cv2.erode(mask, kernel, iterations=iters)
        else:
            raise ValueError(f"지원하지 않는 연산: {op}")
    return mask


def batch_color_filter(
    batch: List[np.ndarray],
    range: List[Tuple[np.ndarray]],
    kernel_size: Tuple[int, int] = (5, 5),
    operations: List[Tuple[str, int]] = [("close", 2), ("open", 2)],
):
    result = []
    for img, (lower, upper) in zip(batch, range):
        result.append(
            color_filtering(
                frame=img,
                lower=lower,
                upper=upper,
                kernel_size=kernel_size,
                operations=operations
            )
        )
    return result
