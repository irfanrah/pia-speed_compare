from typing import List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import Normalize, ToTensor

from ...base import CPModelBase


class ClipEBCOnnx(CPModelBase):
    """ONNX 기반의 CLIP-EBC 군중 계수 모델 클래스.

    슬라이딩 윈도우 방식을 사용하여 크기가 다른 여러 이미지를 효율적으로 처리하고,
    각 이미지의 군중 수를 예측합니다. ONNX Runtime을 통해 최적화된 추론을 수행합니다.

    Attributes:
        session (ort.InferenceSession): 로드된 ONNX 모델의 추론 세션.
        window_size (int): 슬라이딩 윈도우의 크기.
        stride (int): 윈도우의 이동 간격.
        mean (tuple): 이미지 정규화를 위한 평균값.
        std (tuple): 이미지 정규화를 위한 표준편차.
        reduction (int): 원본 이미지와 밀도맵 사이의 크기 감소 비율.
    """

    def __init__(
        self,
        config,
        use_gpu=True,
    ):
        """ClipEBCOnnx 모델 인스턴스를 초기화합니다.

        ONNX 모델 파일을 로드하고, 추론을 위한 실행 프로바이더(GPU 또는 CPU)를 설정합니다.
        또한, 슬라이딩 윈도우 크기, 정규화 파라미터 등 모델에 필요한 설정을 초기화합니다.

        Args:
            config: 모델 설정을 담고 있는 객체. 다음 속성을 포함해야 합니다:
                model_path (str): ONNX 모델 파일의 경로.
                window_size (int): 슬라이딩 윈도우의 크기.
                stride (int): 윈도우의 이동 간격.
                mean (tuple): 이미지 정규화를 위한 채널별 평균.
                std (tuple): 이미지 정규화를 위한 채널별 표준편차.
            use_gpu (bool, optional): GPU 사용 여부를 결정합니다.
                True일 경우 CUDAExecutionProvider를 우선 사용하며,
                사용 불가 시 CPU로 대체됩니다. 기본값은 True입니다.
        """
        self.onnx_model_path = config.model_path
        self.window_size = config.window_size
        self.stride = config.stride
        self.mean = config.mean
        self.std = config.std
        self.reduction = 8
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=self.mean, std=self.std)
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.providers = []
        if use_gpu:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("⚠️ CUDAExecutionProvider not available. Falling back to CPU.")
                self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
        print(f"ONNX 모델 로드 중: {self.onnx_model_path}")
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=self.session_options,
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(
            f"입력 이름: {self.input_name}, 형태: {self.session.get_inputs()[0].shape}"
        )
        print(
            f"출력 이름: {self.output_name}, 형태: {self.session.get_outputs()[0].shape}"
        )
        print(f"실행 제공자: {self.providers}")

    def predict(self, images: List[np.ndarray]) -> List[float]:
        """크기가 다른 여러 이미지에 대한 군중 계수를 일괄 예측합니다.

        입력된 이미지 리스트에서 슬라이딩 윈도우를 통해 모든 이미지 패치를 추출하고,
        이를 하나의 큰 배치로 만들어 ONNX 모델 추론을 한 번에 수행합니다.
        추론 후, 각 이미지에 대한 결과를 재조립하여 최종 인원수를 계산합니다.

        Note:
            입력 리스트의 각 이미지는 크기가 달라도 상관없습니다.

        Args:
            images (List[np.ndarray]): 예측할 이미지들의 리스트. 리스트의 각 요소는
                (H, W, 3) 형태와 `uint8` dtype을 가진 RGB 이미지 Numpy 배열이어야 합니다.
                이미지가 한 개일 경우에도 반드시 리스트에 담아 `[image_np]` 형태로 전달해야 합니다.

        Returns:
            List[float]: 각 이미지에 대한 예측된 군중 수(float)를 담은 리스트.
                입력 이미지 리스트와 동일한 순서를 가집니다.

        Raises:
            TypeError: 리스트에 `np.ndarray`가 아닌 다른 타입의 요소가 포함된 경우.
        """
        all_windows = []
        stitching_info = []

        # TODO : 전처리 이거 한번에 처리 못하나? -> 아이디어가 안떠오름, 이게 최선인가
        ## 아래는 전처리 파트
        for image_np in images:
            # 입력이 항상 (H, W, 3) 형태의 Numpy 배열이라고 가정.
            pil_image = Image.fromarray(image_np)
            tensor_image = self.to_tensor(pil_image)
            normalized_image = self.normalize(tensor_image)
            processed_image = normalized_image.unsqueeze(0).numpy()

            image_height, image_width = processed_image.shape[-2:]

            window_height, window_width = self.window_size, self.window_size
            stride_height, stride_width = self.stride, self.stride

            num_rows = max(
                1, int(np.ceil((image_height - window_height) / stride_height) + 1)
            )
            num_cols = max(
                1, int(np.ceil((image_width - window_width) / stride_width) + 1)
            )

            windows_for_this_image = []
            positions_for_this_image = []

            for r in range(num_rows):
                for c in range(num_cols):
                    x_start, y_start = r * stride_height, c * stride_width

                    final_x_start = (
                        min(x_start, image_height - window_height)
                        if image_height > window_height
                        else 0
                    )
                    final_y_start = (
                        min(y_start, image_width - window_width)
                        if image_width > window_width
                        else 0
                    )
                    final_x_end = final_x_start + window_height
                    final_y_end = final_y_start + window_width

                    window = processed_image[
                        :, :, final_x_start:final_x_end, final_y_start:final_y_end
                    ]
                    windows_for_this_image.append(window)
                    positions_for_this_image.append((x_start, y_start))

            all_windows.extend(windows_for_this_image)
            stitching_info.append(
                {
                    "image_shape": (image_height, image_width),
                    "num_windows": len(windows_for_this_image),
                    "window_positions": positions_for_this_image,
                }
            )

        if not all_windows:
            return [0.0] * len(images)

        # 배치로 추론 시작
        all_windows_np = np.vstack(all_windows)
        ort_inputs = {self.input_name: all_windows_np}
        all_preds = self.session.run([self.output_name], ort_inputs)[0]

        # 결과 재조립 및 계산 ---
        final_counts = []
        preds_cursor = 0
        for info in stitching_info:
            num_windows = info["num_windows"]
            image_preds = all_preds[preds_cursor : preds_cursor + num_windows]
            preds_cursor += num_windows

            image_h, image_w = info["image_shape"]
            pred_h, pred_w = image_h // self.reduction, image_w // self.reduction

            pred_map = np.zeros((1, pred_h, pred_w), dtype=np.float32)
            count_map = np.zeros_like(pred_map)

            for idx, pos in enumerate(info["window_positions"]):
                x_start, y_start = pos
                x_start_out, y_start_out = (
                    x_start // self.reduction,
                    y_start // self.reduction,
                )

                pred_window = image_preds[idx]
                pred_window_h, pred_window_w = pred_window.shape[1:]
                x_end_out, y_end_out = (
                    x_start_out + pred_window_h,
                    y_start_out + pred_window_w,
                )

                x_end_out_clipped, y_end_out_clipped = min(x_end_out, pred_h), min(
                    y_end_out, pred_w
                )

                pred_map[
                    :, x_start_out:x_end_out_clipped, y_start_out:y_end_out_clipped
                ] += pred_window[
                    :,
                    : x_end_out_clipped - x_start_out,
                    : y_end_out_clipped - y_start_out,
                ]
                count_map[
                    :, x_start_out:x_end_out_clipped, y_start_out:y_end_out_clipped
                ] += 1.0

            count_map[count_map == 0] = 1.0
            pred_map /= count_map
            final_counts.append(pred_map.sum())

        return final_counts

    def predict_dense_dot(
        self, images: List[np.ndarray], alpha: float = 0.5
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """
        Args:
            images: List of RGB uint8 numpy arrays
            alpha: 히트맵 투명도 (0~1)
        Returns:
            counts: List[float] 각 이미지 예측 인원 수
            heat_overlays: List[np.ndarray] RGB 원본 위 히트맵 오버레이
            dot_overlays: List[np.ndarray] RGB 원본 위 점 오버레이
        """
        counts = []
        heat_overlays = []
        dot_overlays = []

        # 1) Sliding-window 전처리 및 stitching 정보 수집
        all_windows = []
        stitch_info = []
        for img in images:
            pil = Image.fromarray(img)
            t = self.to_tensor(pil)
            n = self.normalize(t).unsqueeze(0).numpy()
            h, w = n.shape[-2:]
            ws, st = self.window_size, self.stride
            rows = max(1, int(np.ceil((h - ws) / st)) + 1)
            cols = max(1, int(np.ceil((w - ws) / st)) + 1)
            wins, poses = [], []
            for r in range(rows):
                for c in range(cols):
                    x0 = min(r * st, h - ws) if h > ws else 0
                    y0 = min(c * st, w - ws) if w > ws else 0
                    wins.append(n[:, :, x0 : x0 + ws, y0 : y0 + ws])
                    poses.append((x0, y0))
            all_windows.extend(wins)
            stitch_info.append({"shape": (h, w), "num": len(wins), "pos": poses})

        if not all_windows:
            return [], [], []

        # 2) ONNX 배치 추론
        batch = np.vstack(all_windows)
        preds = self.session.run([self.output_name], {self.input_name: batch})[0]

        # 3) 맵 재조립 및 오버레이 생성
        cursor = 0
        for info, orig in zip(stitch_info, images):
            num = info["num"]
            h, w = info["shape"]
            ph, pw = h // self.reduction, w // self.reduction

            # density map 초기화
            dmap = np.zeros((ph, pw), dtype=np.float32)
            norm = np.zeros_like(dmap)
            for i, (x0, y0) in enumerate(info["pos"]):
                pred = preds[cursor + i][0]
                xo, yo = x0 // self.reduction, y0 // self.reduction
                hh, ww = pred.shape
                x1, y1 = min(xo + hh, ph), min(yo + ww, pw)
                dmap[xo:x1, yo:y1] += pred[: x1 - xo, : y1 - yo]
                norm[xo:x1, yo:y1] += 1
            cursor += num
            norm[norm == 0] = 1
            dmap /= norm

            # 인원 수
            total = float(dmap.sum())
            counts.append(total)

            # 원본 해상도로 리사이즈
            d_resized = cv2.resize(dmap, (w, h), interpolation=cv2.INTER_CUBIC)

            # 4) 히트맵 오버레이 (BGR→RGB로 변환)
            heat_u8 = np.uint8(
                255 * (d_resized - d_resized.min()) / max(d_resized.ptp(), 1e-6)
            )
            heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            # 히트맵과 원본 모두 BGR로 처리 후 RGB 변환
            orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
            heat_overlay_bgr = cv2.addWeighted(
                orig_bgr, 1 - alpha, heat_color_bgr, alpha, 0
            )
            heat_overlay = cv2.cvtColor(heat_overlay_bgr, cv2.COLOR_BGR2RGB)
            heat_overlays.append(heat_overlay)

            # 5) 점 오버레이 (BGR→RGB 변환)
            dot_overlay_bgr = orig_bgr.copy()
            dil = cv2.dilate(d_resized, np.ones((3, 3), np.float32))
            ys, xs = np.where(d_resized >= dil)
            coords = list(zip(ys, xs))
            N = int(round(total))
            if len(coords) > N and N > 0:
                values = [d_resized[y, x] for y, x in coords]
                idxs = np.argsort(values)[::-1][:N]
                chosen = [coords[i] for i in idxs]
            else:
                chosen = coords[:N] if N > 0 else []
            for y, x in chosen:
                cv2.circle(
                    dot_overlay_bgr, (x, y), radius=4, color=(0, 255, 0), thickness=-1
                )
            dot_overlay = cv2.cvtColor(dot_overlay_bgr, cv2.COLOR_BGR2RGB)
            dot_overlays.append(dot_overlay)

        return counts, heat_overlays, dot_overlays

    def forward(self, images: List[np.ndarray]) -> Union[float, np.ndarray]:
        return self.predict(images)

    ## 이거 따로 안써서 날림
    @staticmethod
    def _load_model(self):
        pass
