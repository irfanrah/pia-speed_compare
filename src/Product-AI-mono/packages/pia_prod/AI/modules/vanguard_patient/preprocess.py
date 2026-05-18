import cv2
import numpy as np
from typing import Tuple, List
import torch


def enhance_sky_batch(frames: list, **kwargs) -> list:
    """
    여러 프레임이 담긴 리스트를 받아 변환된 리스트를 반환합니다.
    원본 리스트 내의 numpy 배열은 수정되지 않습니다.
    """
    processed_frames = []
    for frame in frames:
        # enhance_sky_vs_blue 함수를 호출하여 새 이미지 생성
        processed = enhance_sky_vs_blue(frame, **kwargs)
        processed_frames.append(processed)
    return processed_frames


def enhance_sky_vs_blue(
    frame: np.ndarray, sat_thresh: int = 255, sat_gamma: float = 2.0, max_value_boost: int = 120
) -> np.ndarray:
    # 1. 원본을 보호하기 위해 내부에서 float32로 변환하며 복사본을 생성합니다.
    # cv2.cvtColor는 원본과 별개의 메모리 공간을 가진 새 배열을 리턴합니다.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    low_mask = s < float(sat_thresh)
    if np.any(low_mask):
        s_orig = s.copy()
        s_ratio = np.clip(s_orig[low_mask] / float(sat_thresh), 0.0, 1.0)
        s[low_mask] = (s_ratio**sat_gamma) * float(sat_thresh)

        boost = (1.0 - s_ratio) * float(max_value_boost)
        v[low_mask] = np.clip(v[low_mask] + boost, 0.0, 255.0)

    hsv_adjusted = cv2.merge((h, s, v)).astype(np.uint8)

    # 2. 최종 결과물 역시 새로운 메모리 객체로 반환됩니다.
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)


def foot_point(bbox_xyxy: List[float]) -> Tuple[int, int]:
    """
    bbox의 하단 중앙 좌표를 반환합니다.
    Args:
        bbox_xyxy: [x1, y1, x2, y2, ...] 형태의 리스트 또는 배열
    Returns:
        (x, y): 하단 중앙 좌표 (정수형)
    """
    x1, y1, x2, y2 = bbox_xyxy[:4]

    # 하단 중앙점 계산
    cx = (x1 + x2) / 2
    by = y2

    return int(cx), int(by)


def is_inside_roi(polygon: np.ndarray, pt: Tuple[int, int]) -> bool:
    """
    점이 다각형 ROI 내부에 있는지 확인합니다.
    Args:
        polygon: ROI 다각형 좌표 (N, 2) 형태의 numpy array
        pt: (x, y) 좌표 튜플
    Returns:
        bool: 내부 또는 경계에 있으면 True, 아니면 False
    """
    # pointPolygonTest: 양수(내부), 0(경계), 음수(외부)
    # measureDist=False로 하면 +1, 0, -1 반환
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def rgb_to_hsv_torch(image: torch.Tensor) -> torch.Tensor:
    """
    PyTorch Tensor (N, 3, H, W) RGB/BGR -> HSV 변환
    입력 범위: [0.0, 1.0]
    출력 범위: [0.0, 1.0]
    """
    # image: (N, 3, H, W)
    max_val, _ = image.max(dim=1, keepdim=True)
    min_val, _ = image.min(dim=1, keepdim=True)
    diff = max_val - min_val

    # 1. Value
    v = max_val

    # 2. Saturation
    s = torch.where(max_val == 0, torch.zeros_like(diff), diff / (max_val + 1e-6))

    # 3. Hue (계산 복잡하므로 필요한 경우 구현하지만, 여기서는 S, V만 쓰므로 생략 가능.
    # 하지만 HSV->RGB 복구를 위해 H도 보존해야 함)
    # Hue 계산은 branch가 많아 마스킹으로 처리
    mask_r = max_val == image[:, 0:1]
    mask_g = max_val == image[:, 1:2]
    mask_b = max_val == image[:, 2:3]

    h = torch.zeros_like(v)

    # r이 max인 경우: (g - b) / diff
    h[mask_r] = (image[:, 1:2][mask_r] - image[:, 2:3][mask_r]) / (diff[mask_r] + 1e-6)
    # g가 max인 경우: 2 + (b - r) / diff
    h[mask_g] = 2.0 + (image[:, 2:3][mask_g] - image[:, 0:1][mask_g]) / (diff[mask_g] + 1e-6)
    # b가 max인 경우: 4 + (r - g) / diff
    h[mask_b] = 4.0 + (image[:, 0:1][mask_b] - image[:, 1:2][mask_b]) / (diff[mask_b] + 1e-6)

    h = (h / 6.0) % 1.0

    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """
    PyTorch Tensor (N, 3, H, W) HSV -> RGB/BGR 변환
    """
    h = hsv[:, 0:1]
    s = hsv[:, 1:2]
    v = hsv[:, 2:3]

    # c = v * s
    # x = c * (1 - torch.abs((h * 6) % 2 - 1))
    # m = v - c

    # zero = torch.zeros_like(c)

    # h 구간에 따른 rgb 값 결정 (조건부 로직을 마스크나 인덱싱으로 처리해야 빠름)
    # 여기서는 간단히 kornia 스타일이나 stack 방식 사용
    # h6 = h * 6
    # r = torch.clamp(torch.abs(h6 - 3) - 1, 0, 1)
    # g = torch.clamp(2 - torch.abs(h6 - 2), 0, 1)
    # b = torch.clamp(2 - torch.abs(h6 - 4), 0, 1)

    # rgb = (torch.cat([r, g, b], dim=1) * s) + (v - v * s)
    # 위 공식은 근사치일 수 있으므로 정확한 공식 적용
    # HSV to RGB 공식: R = V*S * (clamp...) + (V - V*S) = V * (S*... + 1 - S)

    # 더 정확하고 빠른 구현
    i = (h * 6).long()
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i % 6

    # 마스킹 없이 gather 등을 쓸 수도 있지만, 직관적인 조건문 전개
    # (N, 1, H, W)
    out_r = torch.zeros_like(v)
    out_g = torch.zeros_like(v)
    out_b = torch.zeros_like(v)

    mask = i == 0
    out_r[mask] = v[mask]
    out_g[mask] = t[mask]
    out_b[mask] = p[mask]
    mask = i == 1
    out_r[mask] = q[mask]
    out_g[mask] = v[mask]
    out_b[mask] = p[mask]
    mask = i == 2
    out_r[mask] = p[mask]
    out_g[mask] = v[mask]
    out_b[mask] = t[mask]
    mask = i == 3
    out_r[mask] = p[mask]
    out_g[mask] = q[mask]
    out_b[mask] = v[mask]
    mask = i == 4
    out_r[mask] = t[mask]
    out_g[mask] = p[mask]
    out_b[mask] = v[mask]
    mask = i == 5
    out_r[mask] = v[mask]
    out_g[mask] = p[mask]
    out_b[mask] = q[mask]

    return torch.cat([out_r, out_g, out_b], dim=1)


@torch.no_grad()
def enhance_sky_batch_torch(
    images: torch.Tensor, sat_thresh: int = 255, sat_gamma: float = 2.0, max_value_boost: int = 120
) -> torch.Tensor:
    """
    PyTorch Tensor에 대해 Sky Enhancement 수행 (Batch Processing)

    Args:
        images: (N, 3, H, W) Tensor. 값 범위 [0.0, 1.0]이어야 함. (BGR or RGB)
        sat_thresh: 원본 코드의 int (0~255) 기준 임계값
        sat_gamma: 감마 보정 값
        max_value_boost: 원본 코드의 int (0~255) 기준 부스트 값

    Returns:
        Enhanced Images Tensor (N, 3, H, W)
    """
    # 입력이 비었으면 바로 리턴
    if images.numel() == 0:
        return images

    # 1. 파라미터 정규화 (int 0~255 -> float 0.0~1.0)
    # PyTorch Tensor는 보통 0~1로 정규화되어 들어옵니다.
    thresh_float = float(sat_thresh) / 255.0
    boost_float = float(max_value_boost) / 255.0

    # 2. RGB(BGR) -> HSV 변환
    # images는 [0, 1] 범위라고 가정
    hsv = rgb_to_hsv_torch(images)
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

    # 3. Masking (Saturation이 낮은 영역)
    low_mask = s < thresh_float

    # 마스크된 영역이 하나라도 있을 때만 연산 (GPU라서 분기 안타는게 나을수도 있지만, 로직 유지)
    if low_mask.any():
        # S 채널 보정
        # s_ratio = s / thresh
        s_ratio = torch.clamp(s / (thresh_float + 1e-6), 0.0, 1.0)

        # s_new = (s_ratio ** gamma) * thresh
        s_new = (s_ratio**sat_gamma) * thresh_float

        # 마스크 영역만 교체
        s = torch.where(low_mask, s_new, s)

        # V 채널 보정 (Brightness Boost)
        # boost = (1.0 - s_ratio) * max_boost
        boost = (1.0 - s_ratio) * boost_float
        v_new = torch.clamp(v + boost, 0.0, 1.0)

        # 마스크 영역만 교체
        v = torch.where(low_mask, v_new, v)

    # 4. HSV -> RGB(BGR) 복구
    hsv_adjusted = torch.cat([h, s, v], dim=1)
    output = hsv_to_rgb_torch(hsv_adjusted)

    return output
