import numpy as np
import torch
import torch.nn.functional as F


class TorchreidExtractor:
    """ReID embedding extractor using Torchreid; crops via affine_grid + grid_sample."""

    def __init__(
        self,
        weight_path: str,
        arch: str = 'osnet_x0_25',
        img_size=(256, 128),
        device: str = 'cuda',
        fp16: bool = True,
    ):
        import torchreid

        self.device = torch.device(device if isinstance(device, str) else str(device))
        self.fp16 = bool(fp16)
        self.torch_dtype = torch.float16 if self.fp16 else torch.float32
        self.H, self.W = img_size

        state = torch.load(weight_path, map_location='cpu')
        self.model = torchreid.models.build_model(
            name=arch, num_classes=1000, loss='softmax', pretrained=False
        )
        if hasattr(self.model, 'classifier'):
            self.model.classifier = torch.nn.Identity()

        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean.to(self.device).to(self.torch_dtype)
        self.std = std.to(self.device).to(self.torch_dtype)

        if hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = True

        self.use_trt = False
        self.model_trt = None

        # 현재 TRT를 지원히지 않음.
        # try:
        #     if str(self.device).startswith('cuda'):
        #         import tensorrt as trt

        #         scripted = torch.jit.script(self.model.eval().to('cuda'))
        #         dtype_trt = trt.dtype.half if self.fp16 else trt.dtype.float
        #         inputs = [trt.Input((1, 3, self.H, self.W), dtype=dtype_trt)]
        #         self.model_trt = trt.compile(
        #             scripted, inputs=inputs, enabled_precisions={dtype_trt}
        #         )
        #         self.use_trt = True
        # except Exception:
        #     self.use_trt = False
        #     self.model_trt = None

    @torch.inference_mode()
    def __call__(self, image_bgr_np: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if image_bgr_np.ndim != 3 or image_bgr_np.shape[2] != 3:
            raise ValueError("image_bgr_np must be HxWx3 BGR numpy array")

        img = torch.from_numpy(image_bgr_np).to(self.device)
        img = img[..., [2, 1, 0]]
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.to(dtype=self.torch_dtype, non_blocking=True) / (
            255.0 if img.dtype != torch.float32 else 255.0
        )

        Hs, Ws = img.shape[2], img.shape[3]

        dets = torch.as_tensor(boxes_xyxy, device=self.device, dtype=torch.float32)
        N = dets.shape[0]

        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        sx = (x2 - x1).clamp(min=2.0)
        sy = (y2 - y1).clamp(min=2.0)

        theta = torch.zeros((N, 2, 3), device=self.device, dtype=self.torch_dtype)
        sx_d = (sx / float(self.W)).to(self.torch_dtype)
        sy_d = (sy / float(self.H)).to(self.torch_dtype)
        cx_d = ((2.0 * cx / float(Ws)) - 1.0).to(self.torch_dtype)
        cy_d = ((2.0 * cy / float(Hs)) - 1.0).to(self.torch_dtype)

        theta[:, 0, 0] = sx_d
        theta[:, 1, 1] = sy_d
        theta[:, 0, 2] = cx_d
        theta[:, 1, 2] = cy_d

        grid = F.affine_grid(theta, size=(N, 3, self.H, self.W), align_corners=False)
        crops = F.grid_sample(img.expand(N, -1, -1, -1), grid, align_corners=False)

        crops = (crops - self.mean) / self.std

        if self.fp16 and crops.dtype != torch.float16:
            crops = crops.half()
        elif (not self.fp16) and crops.dtype != torch.float32:
            crops = crops.float()

        if self.use_trt and self.model_trt is not None:
            try:
                feats = self.model_trt(crops)
            except Exception:
                out_list = []
                for i in range(crops.shape[0]):
                    with torch.inference_mode():
                        out_list.append(self.model_trt(crops[i : i + 1]))
                feats = torch.cat(out_list, dim=0)
        else:
            feats = self.model(crops)

        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        feats = F.normalize(feats.float(), dim=1)
        return feats.detach().cpu()
