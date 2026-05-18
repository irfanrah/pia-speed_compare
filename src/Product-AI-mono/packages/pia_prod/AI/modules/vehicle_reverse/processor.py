from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from pia_prod.AI.modules.vehicle_reverse.config import (
    OD_CONFIDENCE_THRESHOLD,
    DETECTION_INPUT_SIZE,
    DEVICE,
    MODEL_VEHICLE_REVERSE_TRT_PATH,
    REID_MODEL_PATH,
    TRACKER_CONFIG,
    TRACKER_CONFIG_PATH,
    TRACK_CLASSES,
    WRONGWAY_PARAMS,
    IOU_THRESHOLD,
)
from pia_prod.AI.modules.vehicle_reverse.core_tracking import CoreTracking
from pia_prod.AI.modules.vehicle_reverse.wrongway import (
    LanePolygon,
    WrongWayDetectorPolygon,
    WrongWayParams,
)


class WrongWayProcessor:
    """Processor returning tracking + wrong-way results."""

    def __init__(self, override: Optional[Dict[str, object]] = None):
        override = override or {}
        self.model_path = str(override.get("model_path") or MODEL_VEHICLE_REVERSE_TRT_PATH)
        self.reid_path = str(override.get("reid_path") or REID_MODEL_PATH)
        tracker_yaml_override = override.get("tracker_yaml")
        if tracker_yaml_override:
            self.tracker_yaml = str(tracker_yaml_override)
        else:
            self.tracker_yaml = TRACKER_CONFIG_PATH if TRACKER_CONFIG_PATH else None
        self.tracker_cfg = dict(TRACKER_CONFIG)
        self.tracker_cfg.update(override.get("tracker_cfg", {}))
        self.classes: Optional[Sequence[int]] = override.get("classes") or TRACK_CLASSES
        wrongway_params = dict(WRONGWAY_PARAMS)
        wrongway_params.update(override.get("wrongway_params", {}))
        self.wrongway_params = wrongway_params

        self.tracker: Optional[CoreTracking] = None
        self._stream_lanes: Dict[str, List[LanePolygon]] = {}
        self._stream_wrongway_detectors: Dict[str, WrongWayDetectorPolygon] = {}
        self._init_tracker()

    def _init_tracker(self):
        device = (
            DEVICE
            if DEVICE in {"cuda", "cpu"}
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tracker = CoreTracking(
            model_path=self.model_path,
            tracker_yaml=self.tracker_yaml,
            tracker_cfg=self.tracker_cfg,
            device=device,
            input_size=DETECTION_INPUT_SIZE,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            classes=self.classes,
            reid_weight_path=self.reid_path,
            reid_arch="osnet_ibn_x1_0",
            reid_img_size=(256, 256),
            reid_fp16=device.startswith("cuda"),
            iou_threshold=IOU_THRESHOLD,
        )

    def get_wrongway_for_stream(self, sid: str, roi, direction_vector):
        if sid not in self._stream_wrongway_detectors:
            lanes_local = [LanePolygon(roi, direction_vector)]
            params = WrongWayParams(**self.wrongway_params)
            self._stream_lanes[sid] = lanes_local
            self._stream_wrongway_detectors[sid] = WrongWayDetectorPolygon(lanes_local, params)
        return self._stream_lanes[sid], self._stream_wrongway_detectors[sid]

    def process_frame(
        self,
        im0,
        frame_idx: int,
        stream_id: str,
        now_inference_result,
        now_ratio,
        now_dwdh,
        now_roi,
        now_direction_vector,
    ):
        """Process one image and return per-track results."""
        if self.tracker is None:
            raise RuntimeError("Tracker is not initialized")

        _, wrongway_detector = self.get_wrongway_for_stream(
            stream_id, now_roi, now_direction_vector
        )
        # tracks = self.tracker.track(im0, stream_id=stream_id)
        tracks = self.tracker.track2(now_inference_result, now_ratio, now_dwdh, im0, stream_id, roi=now_roi)

        outputs = []
        for t in tracks:
            cx = t["x"] + t["w"] / 2.0
            cy = t["y"] + t["h"] / 2.0
            wrongway_detector.update(frame_idx, t["id"], cx, cy)
            is_wrong, confidence, _ = wrongway_detector.is_wrong_way(t["id"], cx, cy)
            outputs.append(
                {
                    "frame_idx": frame_idx,
                    "id": int(t["id"]),
                    "bbox": (float(t["x"]), float(t["y"]), float(t["w"]), float(t["h"])),
                    "score": float(t.get("score", 0.0)),
                    "is_wrong": bool(is_wrong),
                    "confidence": float(confidence),
                    "stream_id": stream_id,
                }
            )
        return outputs
