import os

DEVICE = "cuda"
FALL_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_FALL_DETECTION_ONNX_PATH", "assets/model/PersonDet_v3.7.0.onnx"
)
FALL_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_FALL_DETECTION_TRT_PATH", "assets/model/PersonDet_v3.7.0.engine"
)

LIMITED_NUM_OF_FALL_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_FALL_PER_CAMERA", 16))
OD_CONFIDENCE_THRESHOLD = 0.3
OD_NMS_THRESHOLD = 0.35


OD_TIME_INTERVAL_SECOND = 0.3
# 5초 동안 알람 유지 -> 현재 25개의 frame 관측
FALL_QUEUE_SIZE = 5
# 5초 측정동안 1초이상 이벤트 발생시 알람발생 -> 현재 5개의 frame이상시 알람
FALL_QUEUE_THRESHOLD = 3
FALL_CV_CATEGORY = ["fallhazard_cv", "추락위험_cv"]
OD_INPUT_SIZE = (640, 640)
