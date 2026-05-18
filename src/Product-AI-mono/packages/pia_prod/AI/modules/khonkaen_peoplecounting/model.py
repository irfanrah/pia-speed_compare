from pia.ai.tasks.CP.models.clip_ebc.clip_ebc import ClipEBCOnnx
from pia.ai.tasks.CP.base import CPONNXConfig
from typing import List, Union, Any
import torch
import numpy as np
from PIL import Image


class ClipEBCOnnxTorch(ClipEBCOnnx):
    def __init__(self, config: CPONNXConfig | None = None):
        if config is None:
            config = CPONNXConfig()
        super().__init__(config)

    def forward(self, images: Any) -> Union[float, np.ndarray]:
        return self._predict(images)

    @staticmethod
    def _is_HWC(shape):
        if shape[2] == 3:
            return True
        return False

    def _predict(self, images: Any) -> List[float]:
        # 1. 입력 정규화 (단일 -> 리스트)
        input_list = images if isinstance(images, list) else [images]
        if not input_list:
            return []

        all_windows = []
        stitching_info = []

        for img in input_list:
            # --- [Step A] 전처리: Torch 상태 유지 ---
            if isinstance(img, torch.Tensor):
                if img.ndim == 3:  # 지금 구조로는 dim이 3만 오긴함
                    img = (
                        img.permute(2, 0, 1) if img.ndim == 3 and self._is_HWC(img.shape) else img
                    )  # (H, W, C) -> (C, H, W) 변환  # \w ES인풋고려
                    img = img.unsqueeze(0)

                # 정규화: dtype이 정수형이면 255로 나눔 (기존 to_tensor 방식 모사)
                if not img.is_floating_point():
                    img = img.float() / 255.0
                processed_image = self.normalize(img)
            else:
                # Numpy 입력인 경우 (기존 로직 유지)
                pil_image = Image.fromarray(img.astype(np.uint8))
                processed_image = self.normalize(self.to_tensor(pil_image)).unsqueeze(0)

            # --- [Step B] 슬라이딩 윈도우 추출 ---
            _, _, img_h, img_w = processed_image.shape
            win_size = self.window_size
            stride = self.stride

            rows = max(1, int(np.ceil((img_h - win_size) / stride) + 1))
            cols = max(1, int(np.ceil((img_w - win_size) / stride) + 1))

            windows_for_this_image = []
            grid_positions = []  # 기존 코드와의 호환성을 위해 그리드 좌표 저장

            for r in range(rows):
                for c in range(cols):
                    # 1. 원래 그리드 좌표 (재조립용)
                    x_grid, y_grid = r * stride, c * stride

                    # 2. 실제 슬라이싱할 보정 좌표
                    h_start = min(x_grid, img_h - win_size) if img_h > win_size else 0
                    w_start = min(y_grid, img_w - win_size) if img_w > win_size else 0

                    window = processed_image[
                        :, :, h_start : h_start + win_size, w_start : w_start + win_size
                    ]
                    windows_for_this_image.append(window)
                    grid_positions.append((x_grid, y_grid))  # 중요: 보정 전 좌표 저장

            all_windows.extend(windows_for_this_image)
            stitching_info.append(
                {
                    "shape": (img_h, img_w),
                    "num_windows": len(windows_for_this_image),
                    "positions": grid_positions,
                }
            )

        if not all_windows:
            return [0.0] * len(input_list)

        # --- [Step C] ONNX 추론 ---
        combined_tensor = torch.cat(all_windows, dim=0)
        all_windows_np = combined_tensor.detach().cpu().numpy()

        all_preds = self.session.run([self.output_name], {self.input_name: all_windows_np})[0]

        # --- [Step D] 후처리 (기존 로직과 완전히 동일하게 수행) ---
        final_counts = []
        cursor = 0
        res = self.reduction

        for info in stitching_info:
            n = info["num_windows"]
            image_preds = all_preds[cursor : cursor + n]
            cursor += n

            img_h, img_w = info["shape"]
            pred_h, pred_w = img_h // res, img_w // res

            p_map = np.zeros((1, pred_h, pred_w), dtype=np.float32)
            c_map = np.zeros_like(p_map)

            for idx, (x_grid, y_grid) in enumerate(info["positions"]):
                # 기존 로직: 그리드 좌표 기반으로 결과 맵 위치 결정
                h_out_start, w_out_start = x_grid // res, y_grid // res

                pred_window = image_preds[idx]
                win_h, win_w = pred_window.shape[1:]

                h_out_end = min(h_out_start + win_h, pred_h)
                w_out_end = min(w_out_start + win_w, pred_w)

                # 더해질 영역의 크기에 맞춰 슬라이싱 (기존 로직 유지)
                p_map[:, h_out_start:h_out_end, w_out_start:w_out_end] += pred_window[
                    :, : h_out_end - h_out_start, : w_out_end - w_out_start
                ]
                c_map[:, h_out_start:h_out_end, w_out_start:w_out_end] += 1.0

            c_map[c_map == 0] = 1.0  # 0 나누기 방지
            final_counts.append((p_map / c_map).sum())

        return final_counts
