import os

DEVICE = "cuda"
PERSON_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_ONNX_PATH", "assets/model/PersonDet_v3.7.0.onnx"
)
PERSON_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_TRT_PATH", "assets/model/PersonDet_v3.7.0.engine"
)

LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", 16))
OD_CONFIDENCE_THRESHOLD = 0.5
OD_NMS_THRESHOLD = 0.35

OD_TIME_INTERVAL_SECOND = 0.3
# 5초 동안 알람 유지 -> 현재 25개의 frame 관측
PINCH_QUEUE_SIZE = int(os.getenv("PINCH_QUEUE_SIZE", 25))
# 5초 측정동안 1초이상 이벤트 발생시 알람발생 -> 현재 5개의 frame이상시 알람
PINCH_ALARM_DURATION = int(os.getenv("PINCH_ALARM_DURATION", 5))
PINCH_CV_CATEGORY = ["pinch_cv", "협착_cv"]
OD_INPUT_SIZE = (640, 640)
