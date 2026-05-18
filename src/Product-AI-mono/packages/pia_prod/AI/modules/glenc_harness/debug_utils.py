import cv2
import os
import time
import atexit
import signal
import numpy as np
from pathlib import Path
from datetime import datetime
from pia_prod.AI.modules.glenc_harness.config import (
    IMAGE_SAVE_PATH,
    CLS_CONFIDENCE_THRESHOLD,
)

from pia.utils.devtools.debug_tools import save_snapshot


class VideoSaveManager:
    def __init__(
        self,
        output_dir: str,
        save_video: bool,
        model_input_second: float = 1.0,
        save_interval_seconds: int = 60,
        prefix: str = "record",
        fourcc_str: str = "mp4v",
        ext: str = ".mp4",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_video = bool(save_video)
        self.model_input_second = float(model_input_second)
        self.fps = max(1e-6, 1.0 / self.model_input_second)
        self.save_interval_seconds = int(save_interval_seconds)
        self.prefix = prefix
        self.ext = ext

        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.writer = None
        self.curr_path = None
        self.segment_start_ts = None
        self.frame_size = None

        atexit.register(self.close)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        self.close()
        raise SystemExit(0)

    def set_save_video(self, flag: bool):
        flag = bool(flag)
        if self.save_video == flag:
            return
        self.save_video = flag
        if not self.save_video:
            self._close_writer()

    def _new_output_path(self, ts: float) -> Path:
        t = datetime.fromtimestamp(ts)
        fname = f"{self.prefix}_{t.strftime('%Y%m%d_%H%M%S')}{self.ext}"
        return self.output_dir / fname

    def _open_writer_if_needed(self, frame):
        h, w = frame.shape[:2]
        size_changed = self.frame_size is not None and self.frame_size != (w, h)
        if self.writer is None or size_changed:
            if self.writer is not None:
                self._close_writer()
            self.frame_size = (w, h)
            if self.segment_start_ts is None:
                self.segment_start_ts = time.time()
            if self.curr_path is None:
                self.curr_path = self._new_output_path(self.segment_start_ts)
            self.writer = cv2.VideoWriter(
                str(self.curr_path), self.fourcc, self.fps, self.frame_size
            )

    def _rotate_if_needed(self, now_ts: float):
        if self.segment_start_ts is None:
            self.segment_start_ts = now_ts
            self.curr_path = self._new_output_path(now_ts)
            return

        if (now_ts - self.segment_start_ts) >= self.save_interval_seconds:
            self._close_writer()
            self.segment_start_ts = now_ts
            self.curr_path = self._new_output_path(now_ts)

    def write_frame(self, frame, ts: float | None = None):
        if not self.save_video:
            return

        now_ts = ts if ts is not None else time.time()

        self._rotate_if_needed(now_ts)
        self._open_writer_if_needed(frame)

        if self.writer is not None:
            self.writer.write(frame)

    def _close_writer(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.writer = None
        self.curr_path = None

    def close(self):
        self._close_writer()
        self.segment_start_ts = None
        self.frame_size = None


def save_snapshot_for_odcls(
    images,
    stream_ids,
    bboxes,
    raw_cls_results,
    category_index,
    classify_matched_info,
    video_mode,
    video_instance,
):
    classify_count = 0
    for image, stream_id, bboxes_each_image in zip(images, stream_ids, bboxes):
        image = np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy())
        save_dir = os.path.join(IMAGE_SAVE_PATH, stream_id, "harness")

        matched = classify_matched_info.get(stream_id, 0)
        if matched == 0:
            if video_mode and video_instance is not None:
                video_instance.write_frame(image)
            continue

        cls_results = raw_cls_results[classify_count : classify_count + matched]
        is_event = cls_results[:, category_index] >= CLS_CONFIDENCE_THRESHOLD

        for bbox, cls_result in zip(bboxes_each_image, cls_results):
            cls_conf = cls_result[category_index]
            if cls_conf >= CLS_CONFIDENCE_THRESHOLD:
                label, color = "Not wear", (0, 0, 255)
            else:
                label, color = "normal", (255, 0, 0)

            x1, y1, x2, y2 = map(int, bbox[:4])
            od_conf = f"{bbox[4]:.3f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{label} {cls_conf:.2f} person: {od_conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        save_snapshot(image, save_dir=save_dir) if is_event.any() else None

        classify_count += matched

        if video_mode and video_instance is not None:
            video_instance.write_frame(image)
