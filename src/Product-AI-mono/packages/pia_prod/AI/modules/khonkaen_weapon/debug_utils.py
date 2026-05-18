import os
import cv2
import time
import atexit
import signal
import numpy as np
from pathlib import Path
from datetime import datetime
from pia_prod.AI.modules.khonkaen_weapon.config import (
    CATEGORY_DICT,
    CATEGORY_COLOR_DICT,
    IMAGE_SAVE_PATH,
)
from pia.utils.devtools.debug_tools import save_snapshot


class VideoSaveManager:
    def __init__(
        self,
        output_dir: str,
        save_video: bool,
        model_input_second: float = 1.0,  # 프레임 간격(초). FPS = 1 / model_input_second
        save_interval_seconds: int = 60,  # 파일 롤링 주기(초). 영상 저장 주기가 해당 만큼은 아님. 실제 시간(ts)를 반영
        prefix: str = "record",
        fourcc_str: str = "mp4v",  # 코덱: 'mp4v', 'XVID', 'avc1' 등
        ext: str = ".mp4",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_video = bool(save_video)
        self.model_input_second = float(model_input_second)
        self.fps = max(1e-6, 1.0 / self.model_input_second)  # 0 division 방지
        self.save_interval_seconds = int(save_interval_seconds)
        self.prefix = prefix
        self.ext = ext

        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.writer = None
        self.curr_path = None
        self.segment_start_ts = None
        self.frame_size = None  # (w, h)

        # 종료 시 안전하게 닫기
        atexit.register(self.close)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            # 일부 환경(예: Windows 일부 스레드)에서는 시그널 등록이 불가할 수 있음
            pass

    def _signal_handler(self, signum, frame):
        # 안전하게 릴리즈하고 바로 종료
        self.close()
        # 기본 동작 이어가도록
        raise SystemExit(0)

    def set_save_video(self, flag: bool):
        """런타임에 저장 on/off 토글"""
        flag = bool(flag)
        if self.save_video == flag:
            return
        self.save_video = flag
        if not self.save_video:
            # 저장 끌 때 현재 writer 정리
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
            # segment_start_ts는 외부에서 관리, 파일 경로 생성은 _rotate_if_needed에서
            if self.segment_start_ts is None:
                self.segment_start_ts = time.time()
            if self.curr_path is None:
                self.curr_path = self._new_output_path(self.segment_start_ts)
            self.writer = cv2.VideoWriter(
                str(self.curr_path), self.fourcc, self.fps, self.frame_size
            )

    def _rotate_if_needed(self, now_ts: float):
        """세그먼트 경과 시간이 save_interval_seconds를 넘으면 파일 롤링"""
        if self.segment_start_ts is None:
            self.segment_start_ts = now_ts
            self.curr_path = self._new_output_path(now_ts)
            return

        if (now_ts - self.segment_start_ts) >= self.save_interval_seconds:
            # 기존 파일 마감
            self._close_writer()
            # 새 파일로 전환
            self.segment_start_ts = now_ts
            self.curr_path = self._new_output_path(now_ts)

    def write_frame(self, frame, ts: float | None = None):
        """
        외부 서비스 루프에서 매 프레임 호출.
        - save_video가 True일 때만 저장 수행
        - ts가 없으면 time.time() 사용
        """
        if not self.save_video:
            return

        now_ts = ts if ts is not None else time.time()

        # 롤링 여부 확인
        self._rotate_if_needed(now_ts)
        # writer 준비
        self._open_writer_if_needed(frame)

        if self.writer is not None:
            # 컬러/타입 확인(필요 시 변환 추가 가능)
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
        """명시적 종료(서비스 종료 시 호출 권장)"""
        self._close_writer()
        self.segment_start_ts = None
        # frame_size 유지해도 되지만, 다음 세션 안전을 위해 해제
        self.frame_size = None


def save_snapshot_for_weapon(
    total_bbox_info,
    cropped_batches,
    stream_ids,
    alarms,
    video_mode: bool,
    video_instance: VideoSaveManager,
):
    # 디버그 전용 로직
    for batch_idx, bboxes_info in enumerate(total_bbox_info):
        is_now_weapon_event = False
        now_image = cropped_batches[batch_idx]
        now_image = np.ascontiguousarray(now_image.permute(1, 2, 0).cpu().numpy())
        for bbox in bboxes_info:
            x1, y1, x2, y2, conf, cls = bbox
            color = CATEGORY_COLOR_DICT[CATEGORY_DICT[cls]]
            cv2.rectangle(
                img=now_image,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=color,
                thickness=3,
            )
            cv2.putText(
                now_image,
                f"{CATEGORY_DICT[cls]}:{conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            is_now_weapon_event = cls if cls != 0 else is_now_weapon_event
        if len(alarms) > 0 and stream_ids[batch_idx] in alarms:
            if alarms[stream_ids[batch_idx]][0] == 1:
                cv2.putText(
                    now_image,
                    "ALARM!",
                    (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    4,
                )
        if video_mode and video_instance is not None:
            video_instance.write_frame(now_image)
        if is_now_weapon_event:
            save_dir = os.path.join(
                IMAGE_SAVE_PATH,
                stream_ids[batch_idx],
                "weapon",
                CATEGORY_DICT[is_now_weapon_event],
            )
            save_snapshot(image=now_image, save_dir=save_dir)
