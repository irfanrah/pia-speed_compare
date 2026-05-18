import argparse
import math
import os
from glob import glob
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, List, Optional, Tuple, Union
from pia.vision.preprocessing import cv_bgr2rgb_batch
import cv2
from pia_prod.AI.modules.perception_encoder.trt_utils import preprocess_image

from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
from pia_prod.AI.DTO.param_base import RetrievalBase, ROIModel
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.global_config import USER_PARAM_KEY
from pia_prod.AI.modules.perception_encoder.config import (
    CATEGORY_EVENT_MAP,
    INDEX_MAPPING,
    TOP_CANDIDATE,
    TEMPORAL_SIZE,
)
from pia_prod.AI.modules.perception_encoder.service import PEService
from pia_prod.AI.utils.utils import AddStreamModel2dict


class PEDebugService(PEService):
    VIDEO_EXT = (".mp4", ".avi", ".mkv", ".MP4", ".AVI", ".MKV")
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    ANALYSIS_PER_SECOND = 2
    CATEGORY_COLOR = {
        "fire": (0, 0, 255),
        "smoke": (255, 0, 0),
        "falldown": (0, 255, 0),
        "smoking": (255, 0, 255),
        "normal": (200, 200, 200),
    }

    def __init__(self):
        self._analysis_queue = Queue()
        self._category_alias = {
            alias: category for category, aliases in CATEGORY_EVENT_MAP.items() for alias in aliases
        }
        super().__init__(self._analysis_queue)

    # Disable background threads from ServiceBase for offline debugging.
    def thread_ai_inference(self) -> None:
        return

    def thread_run_get_logging_state(self) -> None:
        return

    def send_alarm(self, alarms, batches, stream_ids, user_params, is_needed_cvt_color=False):
        self.last_alarms = alarms

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        # target_txt_vector, target_sentences = self.txtvec_manager.update(
        #   stream_ids, user_params) # FIX : prompts 고정으로 하기로 함

        # batch encode
        # sequnce length 1로 고정
        cv_bgr2rgb_batch(batches)
        cropped_batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        image_cuda = preprocess_image(cropped_batches)
        # torch_batches = torch.from_numpy(np.stack(batches, axis=0)).to(DEVICE)
        visual_vectors = self.model(image_cuda)
        for stream_id, visual_vector in zip(stream_ids, visual_vectors):
            # FIX : image 기반으로 처리하기로 함
            while (
                self.stream_vector_queues[stream_id].__len__() < TEMPORAL_SIZE
            ):  # temporal size 1 로 설정함
                self.stream_vector_queues[stream_id].append(self.zero_mask_vec)
            self.stream_vector_queues[stream_id].append(
                visual_vector
            )  # TODO : 연결 끊긴 스트림은 삭제 해야함

        alarms, predict_infos = self.alarm_event_manager(
            self.stream_vector_queues, self.category_txt_vectors, stream_ids, user_params
        )

        self.send_alarm(alarms, batches, stream_ids, user_params, is_needed_cvt_color=True)

        return alarms, predict_infos

    def analysis_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        top_k: int = 5,
        codec: str = "mp4v",
    ) -> Path:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.stream_vector_queues.clear()
        self.alarm_event_manager.duration_queue.clear()
        self.alarm_event_manager.event_status.clear()
        self.alarm_dict_with_uuid.clear()

        output_root = Path(output_dir) if output_dir else video_path.parent
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{video_path.stem}_debug{video_path.suffix}"

        stream_id = self._make_stream_id(1)
        user_param = self._make_user_param(1)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        source_fps = source_fps if source_fps > 0 else 30.0
        # Throttle inference so we only run analysis roughly twice per second.
        analysis_interval = max(int(round(source_fps / self.ANALYSIS_PER_SECOND)), 1)
        analysis_fps = max(source_fps / analysis_interval, 1.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_analysis_frames = (
            max(math.ceil(total_frames / analysis_interval), 1) if total_frames > 0 else None
        )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, analysis_fps, (width, height))

        active_alarms: Dict[str, str] = {}
        last_predict_info = None
        frame_idx = 0
        analysis_frames_done = 0
        self._report_progress(
            video_path,
            analysis_frames_done,
            total_analysis_frames,
        )

        try:
            while True:
                grabbed = cap.grab()
                if not grabbed:
                    break

                if frame_idx % analysis_interval != 0:
                    frame_idx += 1
                    continue

                ret, frame = cap.retrieve()
                if not ret:
                    break

                model_frame = frame.copy()
                alarms, predict_infos = self._detect(
                    batches=[model_frame],
                    stream_ids=[stream_id],
                    user_params=[user_param],
                )
                self._update_active_alarms(active_alarms, alarms)
                if predict_infos:
                    last_predict_info = predict_infos[0]
                analysis_frames_done += 1
                self._report_progress(
                    video_path,
                    analysis_frames_done,
                    total_analysis_frames,
                )

                annotated = self._draw_overlays(
                    frame,
                    [active_alarms[stream_id]] if stream_id in active_alarms else [],
                    last_predict_info,
                    top_k,
                )
                writer.write(annotated)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        self._report_progress(
            video_path,
            analysis_frames_done,
            total_analysis_frames,
            complete=True,
        )
        return output_path

    def get_video_list(self, dir_path: Union[str, Path]) -> List[str]:
        dir_path = Path(dir_path)
        video_list: List[str] = []
        for ext in self.VIDEO_EXT:
            video_list.extend(glob(os.path.join(str(dir_path), f"*{ext}")))
        return sorted(video_list)

    def analysis_entries(
        self,
        inputs: Iterable[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        top_k: int = 5,
        codec: str = "mp4v",
    ) -> List[Path]:
        video_files: List[Path] = []
        for item in inputs:
            path = Path(item)
            if path.is_dir():
                video_files.extend(Path(p) for p in self.get_video_list(path))
            elif path.is_file():
                video_files.append(path)
            else:
                raise FileNotFoundError(f"Input not found: {path}")

        if not video_files:
            raise ValueError("No video files found for analysis.")

        results = []
        for idx, path in enumerate(sorted(video_files), start=1):
            print(f"[{idx}/{len(video_files)}] Processing {path}")
            results.append(
                self.analysis_video(path, output_dir=output_dir, top_k=top_k, codec=codec)
            )
        return results

    def _make_stream_id(self, index: int) -> str:
        return f"debug_stream_{index}"

    def _make_user_param(self, index: int) -> Dict[str, Dict]:
        ret_events = [
            RetrievalBase(
                name="smoke_ret",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                roi=ROIModel(roiId=1, polygonCoordinates=[]),
                topCandidates=5,
                abnormalText=["smoke"],
                normalText=["normal"],
            ),
            RetrievalBase(
                name="falldown_ret",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                roi=ROIModel(
                    roiId=1,
                    polygonCoordinates=[0, 1871, 1339, 1905, 1346, 358, 83, 355],
                ),
                topCandidates=5,
                abnormalText=["falldown"],
                normalText=["normal"],
            ),
            RetrievalBase(
                name="fire_ret",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                roi=ROIModel(roiId=1, polygonCoordinates=[]),
                topCandidates=5,
                abnormalText=["fire"],
                normalText=["normal"],
            ),
            RetrievalBase(
                name="흡연_ret",
                incidentThresholdSecond=3,
                incidentTimeoutSecond=3,
                roi=ROIModel(roiId=1, polygonCoordinates=[]),
                topCandidates=5,
                abnormalText=["smoking"],
                normalText=["normal"],
            ),
        ]
        add_stream = AddStreamModel(
            cameraId=index,
            cameraUrl=str(index),
            organization="debug",
            retEvent=ret_events,
            timestamp=str_UTC_ISO8601_ms_now_time(),
        )
        return {USER_PARAM_KEY: AddStreamModel2dict(add_stream)}

    def _report_progress(
        self,
        video_path: Union[str, Path],
        processed: int,
        total: Optional[int],
        complete: bool = False,
    ) -> None:
        video_name = Path(video_path).name
        if total is not None:
            remaining = max(total - processed, 0)
            message = f"{video_name}: remaining analysis frames {remaining}/{total}"
        else:
            message = f"{video_name}: analysed {processed} frames"

        prefix = "    -> "
        end = "\n" if complete else "\r"
        print(f"{prefix}{message}", end=end, flush=True)

    def _get_text_render_params(self, frame) -> Dict[str, Union[int, float]]:
        height = frame.shape[0]
        scale = height / 1080.0
        scale = max(0.5, min(3.0, scale))
        font_scale = 0.6 * scale
        thickness = max(1, int(round(scale)))
        margin = max(12, int(round(16 * scale)))
        padding = max(6, int(round(8 * scale)))
        line_gap = max(4, int(round(6 * scale)))
        return {
            "margin": margin,
            "padding": padding,
            "font_scale": font_scale,
            "thickness": thickness,
            "line_gap": line_gap,
        }

    def _update_active_alarms(
        self, active_alarms: Dict[str, str], alarms: Dict[str, Tuple[bool, str]]
    ) -> None:
        if not alarms:
            return
        for stream_id, (flag, category_key) in alarms.items():
            category_name = self._category_alias.get(category_key, category_key)
            if flag is True:
                active_alarms[stream_id] = category_name
            elif flag is False:
                active_alarms.pop(stream_id, None)

    def _draw_overlays(
        self,
        frame,
        active_categories: List[str],
        predict_info,
        top_k: int,
    ):
        render_cfg = self._get_text_render_params(frame)
        if active_categories:
            alarm_entries = [
                (f"Alarm: {category}", self.CATEGORY_COLOR.get(category, (0, 255, 255)))
                for category in active_categories
            ]
            frame = self._draw_text_block(frame, alarm_entries, anchor="tl", **render_cfg)

        prompt_char_limit = max(32, int(round(48 * (render_cfg["font_scale"] / 0.6))))
        topk_entries = self._format_topk_entries(
            predict_info,
            top_k,
            vis_prompt_length=prompt_char_limit,
        )
        if topk_entries:
            frame = self._draw_text_block(frame, topk_entries, anchor="tr", **render_cfg)
        return frame

    def _format_topk_entries(
        self, predict_info, top_k: int, vis_prompt_length=70
    ) -> List[Tuple[str, Tuple[int, int, int]]]:
        if not predict_info:
            return []
        sims, classes, prompts = predict_info

        if hasattr(sims, "detach"):
            sims = sims.detach().cpu().numpy()
        sims = sims.tolist() if hasattr(sims, "tolist") else list(sims)
        classes = classes.tolist() if hasattr(classes, "tolist") else list(classes)
        prompts = prompts.tolist() if hasattr(prompts, "tolist") else list(prompts)

        entries: List[Tuple[str, Tuple[int, int, int]]] = []
        for score, cls_id, prompt in zip(sims, classes, prompts):
            if len(entries) >= top_k:
                break
            class_name = INDEX_MAPPING.get(int(cls_id), str(cls_id))
            color = self.CATEGORY_COLOR.get(class_name, self.CATEGORY_COLOR["normal"])
            prompt = str(prompt)
            if len(prompt) > vis_prompt_length:
                prompt = f"{prompt[:vis_prompt_length-3]}..."
            entries.append((f"{class_name}: {score:.3f} | {prompt}", color))
        return entries

    def _draw_text_block(
        self,
        frame,
        entries: List[Tuple[str, Tuple[int, int, int]]],
        anchor: str,
        margin: int = 16,
        padding: int = 8,
        font_scale: float = 0.6,
        thickness: int = 1,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.4,
        line_gap: int = 6,
    ):
        if not entries:
            return frame

        sizes: List[Tuple[int, int]] = []
        baselines: List[int] = []
        for text, _ in entries:
            (w_text, h_text), baseline = cv2.getTextSize(text, self.FONT, font_scale, thickness)
            sizes.append((w_text, h_text))
            baselines.append(baseline)

        text_heights = [h + b for (_, h), b in zip(sizes, baselines)]
        box_width = max(w for w, _ in sizes) + padding * 2
        box_height = (
            padding * 2 + sum(text_heights) + line_gap * (len(entries) - 1 if entries else 0)
        )

        h, w = frame.shape[:2]
        if anchor == "tr":
            top_left = (w - margin - box_width, margin)
        else:
            top_left = (margin, margin)
        bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        y = top_left[1] + padding
        for (text, color), (_, h_text), baseline in zip(entries, sizes, baselines):
            y += h_text
            cv2.putText(
                frame,
                text,
                (top_left[0] + padding, y),
                self.FONT,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += baseline + line_gap
        return frame


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate debug videos with Perception Encoder predictions."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video file paths or directories containing video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store annotated videos. Defaults to input location.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=TOP_CANDIDATE,
        help="Number of Top-K prompts to overlay.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec to use for the output video.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    service = PEDebugService()
    results = service.analysis_entries(
        inputs=args.inputs,
        output_dir=args.output_dir,
        top_k=args.topk,
        codec=args.codec,
    )
    print("\nSaved annotated videos:")
    for path in results:
        print(f" - {path}")


if __name__ == "__main__":
    main()
