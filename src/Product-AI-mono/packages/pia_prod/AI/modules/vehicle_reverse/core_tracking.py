import yaml
import torch
import numpy as np
import cv2
from typing import Optional, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from types import SimpleNamespace

from .reid_extractor import TorchreidExtractor


class CoreTracking:
    """Object tracking using YOLO + BoT-SORT."""

    def __init__(
        self,
        model_path,
        tracker_yaml=None,
        tracker_cfg=None,
        device=None,
        roi_polygons=None,
        conf_thres=0.25,
        classes=None,
        reid_weight_path: str = None,
        reid_arch: str = 'osnet_x0_25',
        reid_img_size=(256, 128),
        reid_fp16: bool = True,
        input_size: Tuple[int, int] = (640, 640),
        iou_threshold: float = 0.6,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path, task="detect")
        self._predict_device = self._resolve_predict_device(self.device)

        self.conf_thres = conf_thres
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.roi_polygons = roi_polygons or []
        self.input_size = tuple(input_size) if input_size else (640, 640)

        self.reid_extractor = None
        if reid_weight_path:
            self.reid_extractor = TorchreidExtractor(
                weight_path=reid_weight_path,
                arch=reid_arch,
                img_size=reid_img_size,
                device=self.device,
                fp16=reid_fp16 and (self.device.startswith('cuda')),
            )

        self.tracker_yaml = tracker_yaml
        self.tracker_cfg = tracker_cfg or {}
        self.trackers = {}
        default_tracker = self._init_tracker(
            self.device, self.tracker_yaml, self.tracker_cfg, with_reid=bool(self.reid_extractor)
        )
        self.trackers[None] = default_tracker
        self.predict_kwargs = dict(
            conf=self.conf_thres,
            verbose=False,
            classes=self.classes,
            iou=self.iou_threshold,
        )
        if self._predict_device is not None:
            self.predict_kwargs["device"] = self._predict_device

    @staticmethod
    def _point_in_poly(px, py, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cond = ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1)
            if cond:
                inside = not inside
        return inside

    @staticmethod
    def _init_tracker(device, tracker_yaml, tracker_cfg=None, with_reid=True):
        default_args = dict(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=0.8,
            max_age=30,
            max_iou_distance=0.2,
            with_reid=with_reid,
            device=device,
            half=False,
            model="auto",
        )
        cfg = dict(default_args)
        if tracker_yaml and isinstance(tracker_yaml, str):
            try:
                with open(tracker_yaml, "r") as f:
                    y = yaml.safe_load(f) or {}
                cfg.update({k: v for k, v in y.items() if v is not None})
            except Exception:
                pass
        if tracker_cfg:
            cfg.update({k: v for k, v in tracker_cfg.items() if v is not None})
        cfg_obj = SimpleNamespace(**cfg)
        tracker = BOTSORT(args=cfg_obj)
        tracker.reset()
        return tracker

    @staticmethod
    def _letterbox(image: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)):
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        shape = image.shape[:2]  # h, w
        if not shape[0] or not shape[1]:
            raise ValueError("Invalid image shape for letterbox")

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2.0
        dh = (new_shape[0] - new_unpad[1]) / 2.0

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        ratio = (new_unpad[0] / shape[1], new_unpad[1] / shape[0])
        return image, ratio, (dw, dh)

    @staticmethod
    def _resolve_predict_device(device: Optional[str]):
        if device is None:
            return None
        dev = str(device).lower()
        if dev.startswith("cuda"):
            if ":" in dev:
                _, idx = dev.split(":", 1)
                try:
                    return int(idx)
                except ValueError:
                    return dev
            return 0
        if dev in {"cpu", "mps"}:
            return dev
        return dev

    @staticmethod
    def parse_track_array(a):
        arr = np.asarray(a).reshape(-1).astype(float)
        if arr.size < 6:
            raise ValueError(f"Track array too short: {arr}")
        x1, y1, x2, y2 = arr[0], arr[1], arr[2], arr[3]
        if x2 > x1 and y2 > y1:
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
        else:
            x, y, w, h = x1, y1, max(0.0, x2), max(0.0, y2)
        tid = int(arr[4])
        score = float(arr[5])
        return x, y, w, h, tid, score

    def _filter_boxes(self, boxes, im_shape, roi):
        if boxes is None or len(boxes) == 0:
            empty = torch.zeros((0, 6), dtype=torch.float32, device="cpu")
            return Boxes(empty, orig_shape=im_shape)

        xyxy = boxes.xyxy
        conf = boxes.conf.reshape(-1)
        if hasattr(boxes, "cls") and boxes.cls is not None:
            cls = boxes.cls.reshape(-1)
        else:
            cls = torch.zeros_like(conf, dtype=torch.int64)

        if self.classes is None:
            keep_cls = torch.ones_like(conf, dtype=torch.bool)
        else:
            try:
                keep_cls = torch.isin(
                    cls.to(torch.int64),
                    torch.tensor(self.classes, device=cls.device, dtype=torch.int64),
                )
            except AttributeError:
                keep_cls = torch.from_numpy(
                    np.isin(
                        cls.detach().cpu().numpy().astype(int), np.array(self.classes, dtype=int)
                    )
                ).to(torch.bool)

        if len(roi) == 0:
            keep_roi = torch.ones_like(conf, dtype=torch.bool)
        else:
            centers = ((xyxy[:, :2] + xyxy[:, 2:4]) * 0.5).detach().cpu().numpy()
            keep_roi_np = np.zeros((centers.shape[0],), dtype=bool)
            poly_np = np.asarray(roi, dtype=float)
            min_xy = poly_np.min(axis=0)
            max_xy = poly_np.max(axis=0)
            in_bbox = (
                (centers[:, 0] >= min_xy[0])
                & (centers[:, 0] <= max_xy[0])
                & (centers[:, 1] >= min_xy[1])
                & (centers[:, 1] <= max_xy[1])
            )
            if in_bbox.any():
                for idx in np.nonzero(in_bbox)[0]:
                    px, py = centers[idx]
                    if self._point_in_poly(px, py, roi):
                        keep_roi_np[idx] = True

            keep_roi = torch.from_numpy(keep_roi_np).to(device=conf.device)

        keep = keep_cls & keep_roi
        if keep.any():
            t_xyxy = xyxy[keep].to(torch.float32)
            t_conf = conf[keep].to(torch.float32).unsqueeze(1)
            t_cls = cls[keep].to(torch.float32).unsqueeze(1)
            t = torch.cat([t_xyxy, t_conf, t_cls], dim=1)
            return Boxes(t, orig_shape=im_shape)

        empty = torch.zeros((0, 6), dtype=torch.float32, device="cpu")
        return Boxes(empty, orig_shape=im_shape)

    def _extract_feats_for_boxes(
        self, image_bgr_np: np.ndarray, filtered_boxes: Boxes
    ) -> np.ndarray:
        if self.reid_extractor is None:
            return None
        if len(filtered_boxes) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        xyxy = filtered_boxes.xyxy.detach().cpu().numpy()
        feats = self.reid_extractor(image_bgr_np, xyxy)
        return feats

    def _tracker_update(self, det_boxes: Boxes, image_bgr_np: np.ndarray, feats: np.ndarray):
        if isinstance(det_boxes, tuple):
            tracker, det_boxes, image_bgr_np, feats = det_boxes
        raise RuntimeError(
            "_tracker_update requires tracker instance; call the new signature in code paths"
        )

    def _tracker_supports_feats(self, tracker):
        import inspect

        sig = inspect.signature(tracker.update)
        return len(sig.parameters) >= 3  # self 제외 2개 → dets, image

    def _tracker_update_for(self, tracker, det_boxes, image_bgr_np, feats):
        # 1회만 체크
        if not hasattr(tracker, "_supports_feats"):
            tracker._supports_feats = self._tracker_supports_feats(tracker)

        # feats 지원 안 하면 feats를 절대 넣지 말기
        if not tracker._supports_feats:
            return tracker.update(det_boxes, image_bgr_np)

        # feats 지원한다고 판단될 때만
        return tracker.update(det_boxes, image_bgr_np, feats)

    def track(self, image, stream_id: str):
        letterboxed, ratio, dwdh = self._letterbox(image, self.input_size)
        results = self.model(letterboxed, **self.predict_kwargs)
        r = results[0]
        boxes: Boxes = r.boxes
        orig_h, orig_w = image.shape[:2]
        if boxes is not None and len(boxes):
            # Clone/detach because ultralytics may return inference tensors that disallow inplace ops
            boxes_xyxy = boxes.xyxy.detach().clone()
            dwdh_tensor = torch.tensor(
                [dwdh[0], dwdh[1], dwdh[0], dwdh[1]],
                device=boxes_xyxy.device,
                dtype=boxes_xyxy.dtype,
            )
            ratio_tensor = torch.tensor(
                [ratio[0], ratio[1], ratio[0], ratio[1]],
                device=boxes_xyxy.device,
                dtype=boxes_xyxy.dtype,
            )
            boxes_xyxy -= dwdh_tensor
            boxes_xyxy /= ratio_tensor
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp_(0, orig_w)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp_(0, orig_h)
            boxes_data = boxes.data.detach().clone()
            boxes_data[:, :4] = boxes_xyxy
            new_boxes = Boxes(boxes_data, orig_shape=image.shape[:2])
            if hasattr(boxes, "names"):
                setattr(new_boxes, "names", getattr(boxes, "names"))
            boxes = new_boxes

        filtered = self._filter_boxes(boxes, image.shape[:2])
        filtered_cpu = filtered.cpu()
        feats = self._extract_feats_for_boxes(image, filtered)
        tracker = self.trackers.get(stream_id)
        if tracker is None:
            tracker = self._init_tracker(
                self.device,
                self.tracker_yaml,
                self.tracker_cfg,
                with_reid=bool(self.reid_extractor),
            )
            self.trackers[stream_id] = tracker
        online_targets = self._tracker_update_for(tracker, filtered_cpu, image, feats)

        tracks = []
        for t in online_targets:
            x, y, w, h, tid, score = self.parse_track_array(t)
            tracks.append(
                {"x": x, "y": y, "w": w, "h": h, "id": tid, "score": score, 'stream_id': stream_id}
            )
        return tracks

    def track2(self, inference_result, ratio, dwdh, image, stream_id, roi):
        r = inference_result
        boxes: Boxes = r.boxes
        orig_h, orig_w = image.shape[:2]
        if boxes is not None and len(boxes):
            # Clone/detach because ultralytics may return inference tensors that disallow inplace ops
            boxes_xyxy = boxes.xyxy.detach().clone()
            dwdh_tensor = torch.tensor(
                [dwdh[0], dwdh[1], dwdh[0], dwdh[1]],
                device=boxes_xyxy.device,
                dtype=boxes_xyxy.dtype,
            )
            ratio_tensor = torch.tensor(
                [ratio[0], ratio[1], ratio[0], ratio[1]],
                device=boxes_xyxy.device,
                dtype=boxes_xyxy.dtype,
            )
            boxes_xyxy -= dwdh_tensor
            boxes_xyxy /= ratio_tensor
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp_(0, orig_w)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp_(0, orig_h)
            boxes_data = boxes.data.detach().clone()
            boxes_data[:, :4] = boxes_xyxy
            new_boxes = Boxes(boxes_data, orig_shape=image.shape[:2])
            if hasattr(boxes, "names"):
                setattr(new_boxes, "names", getattr(boxes, "names"))
            boxes = new_boxes

        filtered = self._filter_boxes(boxes, image.shape[:2], roi=roi)
        filtered_cpu = filtered.cpu()
        feats = self._extract_feats_for_boxes(image, filtered)
        tracker = self.trackers.get(stream_id)
        if tracker is None:
            tracker = self._init_tracker(
                self.device,
                self.tracker_yaml,
                self.tracker_cfg,
                with_reid=bool(self.reid_extractor),
            )
            self.trackers[stream_id] = tracker
        online_targets = self._tracker_update_for(tracker, filtered_cpu, image, feats)

        tracks = []
        for t in online_targets:
            x, y, w, h, tid, score = self.parse_track_array(t)
            tracks.append(
                {"x": x, "y": y, "w": w, "h": h, "id": tid, "score": score, 'stream_id': stream_id}
            )
        return tracks
