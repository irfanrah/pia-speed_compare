import cv2
import numpy as np


def get_sequence(cap: cv2.VideoCapture, start_frame_id: int, sequence_length: int) -> np.ndarray:
    # Get height, width and channel of the video
    ret, frame = cap.read()
    assert ret
    height, width, channel = frame.shape

    # Get sequence
    sequence = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    for i in range(sequence_length):
        ret, frame = cap.read()
        assert ret

        sequence.append(frame)

    sequence_np = np.asarray(sequence)
    assert sequence_np.shape == (sequence_length, height, width, channel)

    return sequence_np
