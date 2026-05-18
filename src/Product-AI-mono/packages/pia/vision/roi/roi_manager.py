from abc import ABC
from typing import List, Tuple, Union, final

import cv2
import numpy as np
import torch
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import calc_expand_coord


class ROIManagerBase(ABC):
    @final
    def erase_roi(self, batch, rois):
        """
        주어진 batch, roi 짝에 대해서 roi 영역 내부에 해당하는 영역을 검은색으로 변경 후 리턴
        """
        return apply_resized_batch_frames_erase_roi(batch, rois)

    @final
    def crop_roi(self, batch, rois):
        """
        주어진 batch, 다각형 roi 짝에 대해서 roi 영역만 crop하여 직사각형으로 리턴
        (다각형에서 직사각형으로 변환 시 빈 공간은 검은색으로 처리)
        """
        # TODO : 코드 작성 필요
        pass

    @staticmethod
    def get_pair_list(input_list):
        """
        1차원 리스트를 2차원 리스트로 변환하여 좌표 쌍을 만든다.

        Args:
            input_list (list): [x1, y1, x2, y2, ..., xn, yn] 형태의 1차원 리스트.

        Returns:
            np.array: [[x1, y1], [x2, y2], ..., [xn, yn]] 형태의 2차원 numpy 배열.

        Raises:
            ValueError: 리스트 길이가 홀수인 경우 예외 발생.
        """
        if len(input_list) % 2 != 0:
            raise ValueError("Input list length must be even.")

        return np.array(
            [input_list[i : i + 2] for i in range(0, len(input_list), 2)], dtype=np.int32
        )
        
    @staticmethod
    def clip_roi(roi, w, h):
        """
        ROI 좌표가 이미지 경계를 벗어나는 경우, 경계 내로 클리핑합니다.

        Parameters
        ----------
        roi : array-like
            [(x1, y1), (x2, y2), …] 형태로 주어지는 다각형 꼭짓점 좌표.
        frame_wh : tuple
            (width, height) 형태의 이미지 크기.

        Returns
        -------
        np.ndarray
            클리핑된 ROI 좌표를 포함하는 배열.
        """
        roi = np.array(roi, dtype=np.int32)
        roi[:, 0] = np.clip(roi[:, 0], 0, w - 1)
        roi[:, 1] = np.clip(roi[:, 1], 0, h - 1)
        return roi


def apply_erase_roi(frame: np.ndarray, roi):
    """
    주어진 이미지에서 다각형(ROI) 내부 픽셀을 모두 0으로 지웁니다.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, C) 또는 (H, W) 형태의 이미지 배열.
    roi : array-like
        [(x1, y1), (x2, y2), …] 형태로 주어지는 다각형 꼭짓점 좌표.

    Returns
    -------
    np.ndarray
        ROI 영역이 검게(0) 처리된 새 이미지.
    """
    # 원본 보존
    out = frame.copy()
    if len(roi) == 0:
        return out

    # ROI 마스크 생성
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.asarray(roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    # 다각형 내부 픽셀 0으로 지우기
    out[mask == 255] = 0
    return out


def apply_batch_frames_erase_roi(batch, rois):
    result = []
    for frame, roi in zip(batch, rois):
        if len(roi) == 0:
            result.append(frame)
            continue
        roi_erased_frame = apply_erase_roi(frame, roi)
        result.append(roi_erased_frame)
    return result


def fast_letterbox_with_roi(
    img: np.ndarray,
    roi_pts: List[Tuple[int, int]],
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
):
    """
    • cv2.warpAffine 한 번으로 리사이즈+패딩
    • ROI 좌표는 (N,2) 배열로 벡터화 변환
    • 출력: resized_img (np.ndarray), new_roi (List[Tuple[int,int]])
    """
    h, w = img.shape[:2]
    tgt_w, tgt_h = target_size

    # 스케일·패딩 계산
    scale = min(tgt_w / w, tgt_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_left = (tgt_w - new_w) // 2
    pad_top = (tgt_h - new_h) // 2

    # ── ① 이미지 변환: resize + pad  (단일 패스)
    M = np.array([[scale, 0, pad_left], [0, scale, pad_top]], dtype=np.float32)
    resized = cv2.warpAffine(
        img,
        M,
        (tgt_w, tgt_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=pad_color,
    )

    # ── ② ROI 좌표 벡터화 변환
    if roi_pts:
        pts = np.asarray(roi_pts, dtype=np.float32)  # (N,2)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)  # (N,1)
        homo = np.hstack([pts, ones])  # (N,3)
        new = homo @ M.T  # (N,2)
        new_roi = np.round(new).astype(int).tolist()
    else:
        new_roi = []

    return resized, new_roi


def letterbox_resize_with_roi(
    img: np.ndarray,
    roi_pts: List[Tuple[int, int]],
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    • 원본 이미지 종횡비를 유지한 채 640×640(기본값)로 리사이즈하고
    • 부족한 영역은 pad_color(기본 (114,114,114))로 패딩
    • ROI 좌표를 새 이미지 기준으로 변환해 반환

    Parameters
    ----------
    img : np.ndarray
        원본 BGR 이미지 (H, W, 3)
    roi_pts : list[int] | list[tuple[int,int]]
        [x1,y1,x2,y2, …] 또는 [(x1,y1),(x2,y2)…] 형식의 다각형 좌표
    target_size : (int, int), default (640,640)
        (width, height)  원하는 출력 크기
    pad_color : (int,int,int), default (114,114,114)
        패딩 색 (B,G,R)

    Returns
    -------
    resized_img : np.ndarray
        패딩까지 완료된 640×640 이미지
    new_roi : list[tuple[int,int]]
        리사이즈·패딩 이후의 ROI 꼭짓점 좌표
    """
    # --- ROI를 (N,2) 형태로 변환 ------------------------------------------------
    # --- 원본 크기, 스케일, 패딩 계산 -------------------------------------------
    h, w = img.shape[:2]
    tgt_w, tgt_h = target_size
    scale = min(tgt_w / w, tgt_h / h)

    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_w, pad_h = tgt_w - new_w, tgt_h - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2

    # --- 리사이즈 & 패딩 --------------------------------------------------------
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_img = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    # --- ROI 좌표 변환 ----------------------------------------------------------
    if len(roi_pts):
        new_roi = [
            (
                int(round(x * scale + pad_left)),
                int(round(y * scale + pad_top)),
            )
            for x, y in roi_pts
        ]
    else:
        new_roi = []
    return resized_img, new_roi


def get_resized_and_erased_frame(frame, roi):
    resized_img, new_roi = fast_letterbox_with_roi(frame, roi)
    erased_roi_resized_img = apply_erase_roi(resized_img, new_roi)
    return erased_roi_resized_img, new_roi


def apply_resized_batch_frames_erase_roi(batch, rois):
    resized_erased_frames = []
    new_rois = []

    for frame, roi in zip(batch, rois):
        roi_erased_frame, new_roi = get_resized_and_erased_frame(frame, roi)
        resized_erased_frames.append(roi_erased_frame)
        new_rois.append(new_roi)
    return resized_erased_frames, new_rois


def calc_letterbox_parameter(source_shape, target_shape=(640, 640)):
    """
    원본 이미지의 비율을 유지하면서 target 크기로 맞추기 위한 letterbox padding 값을 계산한다.

    Args:
        source_shape (tuple): 원본 이미지 크기 (height, width).
        target_shape (tuple, optional): 변환할 목표 크기 (width, height). 기본값은 (640, 640).

    Returns:
        tuple: (배율 r, (좌우 padding, 상하 padding))
                - r: 원본 이미지를 target 크기에 맞추기 위한 스케일링 비율.
                - (dw, dh): 좌우(dw), 상하(dh) padding 값.

    Example:
        >>> source_shape = (1080, 1920)
        >>> target_shape = (640, 640)
        >>> calc_letterbox_parameter(source_shape, target_shape)
        (0.333, (0, 0))  # 비율 0.333, 좌우/상하 padding 없음
    """
    r = min(target_shape[0] / source_shape[0], target_shape[1] / source_shape[1])
    r = min(r, 1.0)
    new_unpad = int(round(source_shape[1] * r)), int(round(source_shape[0] * r))
    dw, dh = (target_shape[1] - new_unpad[0]) // 2, (
        target_shape[0] - new_unpad[1]
    ) // 2  # wh padding
    return r, (dw, dh)


def calc_expand_roi(batch_shape, roi):  # roi
    height, width, _ = batch_shape
    rois = np.array(roi, dtype=np.int32)
    expanded_roi = calc_expand_coord(roi=rois, frame_wh=(width, height), expand_ratio=0.3)
    re_calc_origin_roi = rois - expanded_roi[0]
    expanded_roi_shape = list(reversed(list(expanded_roi[2] - expanded_roi[0])))
    r, pad = calc_letterbox_parameter(source_shape=expanded_roi_shape)
    after_letterbox_calc_origin_roi = ((re_calc_origin_roi * r) + np.array(pad)).astype(np.int32)
    return expanded_roi, after_letterbox_calc_origin_roi


def cv_crop_region(frame: np.array, region: list):
    if len(region) > 0:
        region = np.array(region)
        xmin, xmax, ymin, ymax = (
            region[:, 0].min(),
            region[:, 0].max(),
            region[:, 1].min(),
            region[:, 1].max(),
        )
        h, w = frame.shape[:2]
        mask = np.zeros([h, w], dtype=np.uint8)
        cv2.drawContours(
            mask,
            [np.array(region, dtype=np.int32)],
            -1,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        dst = cv2.bitwise_and(frame, frame, mask=mask)
        dst = dst[ymin:ymax, xmin:xmax, :]
    else:
        dst = frame
    return dst


def _to_chw(frame: torch.Tensor) -> torch.Tensor:
    # TODO : util 함수 이관 필요
    if frame.ndim != 3:
        raise ValueError("frame must be 3D.")
    if frame.shape[0] in (1, 3, 4):   # (C,H,W)
        return frame.contiguous()
    if frame.shape[-1] in (1, 3, 4):  # (H,W,C) -> (C,H,W)
        return frame.permute(2, 0, 1).contiguous()
    raise ValueError("frame must be (C,H,W) or (H,W,C).")


def torch_crop_region(frame: torch.Tensor, poly_np: np.array, pad_color: Tuple[int] = (114, 114, 114)) -> torch.Tensor:
    chw = _to_chw(frame)
    C, H, W = chw.shape
    device, dtype = chw.device, chw.dtype

    # --- polygon & bbox
    poly = np.asarray(poly_np, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("region polygon must have shape (N,2)")

    x_min = int(np.floor(max(0, poly[:, 0].min())))
    y_min = int(np.floor(max(0, poly[:, 1].min())))
    x_max = int(np.ceil(min(float(W - 1), poly[:, 0].max())))
    y_max = int(np.ceil(min(float(H - 1), poly[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        # 비정상 ROI → 빈 텐서 반환
        return frame

    # --- 원본과 분리된 crop 텐서 (작은 영역만 clone이므로 오버헤드 작음)
    crop = chw[:, y_min:y_max + 1, x_min:x_max + 1].clone()   # (C,h,w)
    h, w = crop.shape[1], crop.shape[2]

    # --- bbox 좌표계의 polygon (torch로)
    poly_x = torch.as_tensor(poly[:, 0] - x_min, device=device, dtype=torch.float32)
    poly_y = torch.as_tensor(poly[:, 1] - y_min, device=device, dtype=torch.float32)
    n = poly_x.numel()

    # --- bbox grid
    xs = torch.arange(w, device=device, dtype=torch.float32)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    px = grid_x.reshape(-1)
    py = grid_y.reshape(-1)

    # --- ray casting (even-odd): mask=True → 다각형 내부
    inside = torch.zeros_like(px, dtype=torch.bool)
    xj, yj = poly_x[-1], poly_y[-1]
    eps = 1e-12
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + eps) + xi)
        inside ^= cond
        xj, yj = xi, yi
    mask = inside.reshape(h, w)  # True=ROI 내부

    # --- ROI 바깥(~mask)만 114로 채우기
    bg = torch.tensor(pad_color, device=device, dtype=dtype)
    if C == 1:
        crop[0][~mask] = bg[0]
    else:
        # RGB까지만 채움(필요 시 range(C)로 변경)
        for c in range(min(C, 3)):
            crop[c][~mask] = bg[c]

    return crop


@torch.inference_mode()
def batch_crop_region(
    frames: Union[List[torch.Tensor], List[np.ndarray]],
    regions: List[np.ndarray],
    pad_color: Tuple[int] = (114, 114, 114)
) -> List[Union[torch.Tensor, np.ndarray]]:
    """
    각 (frame, polygon ROI) 쌍에 대해:
      - torch.Tensor 입력이면 torch_crop_region 사용
      - np.ndarray 입력이면 cv_crop_region 사용
    반환: cropped_frames (원본 프레임은 변경되지 않음)
    """
    assert len(frames) == len(regions), "frames와 regions 길이가 같아야 합니다."
    cropped_frames = []

    if isinstance(frames[0], torch.Tensor):
        # Torch 버전
        for frame, poly_np in zip(frames, regions):
            crop = torch_crop_region(frame, poly_np, pad_color=pad_color)
            cropped_frames.append(crop)
    elif isinstance(frames[0], np.ndarray):
        # OpenCV (numpy) 버전
        for frame, poly_np in zip(frames, regions):
            crop = cv_crop_region(frame, poly_np)
            cropped_frames.append(crop)
    else:
        raise TypeError("frames는 torch.Tensor 리스트 또는 np.ndarray 리스트여야 합니다.")

    return cropped_frames
