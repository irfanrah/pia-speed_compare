import os
import tempfile

import cv2
import numpy as np
from pia.vision.load.opencv import get_sequence


def _create_dummy_video(path: str, num_frames: int = 5, size: int = 4) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 1.0, (size, size))
    for i in range(num_frames):
        frame = np.full((size, size, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_get_sequence_reads_expected_frames():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        filename = tmp.name
    try:
        _create_dummy_video(filename)
        cap = cv2.VideoCapture(filename)
        sequence = get_sequence(cap, start_frame_id=1, sequence_length=3)
        cap.release()

        assert sequence.shape == (3, 4, 4, 3)
        assert not np.all(sequence[0])  # 모두 0인 프레임
        assert not np.all(sequence[2])  # 0과 4로 이루어진 프레임
    finally:
        os.remove(filename)
