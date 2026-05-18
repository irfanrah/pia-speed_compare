import os

DEVICE = "cuda"
HARNESS_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_HARNESS_DETECTION_ONNX_PATH", "assets/model/HarnessOffDet_v0.0.0.onnx"
)
HARNESS_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_HARNESS_DETECTION_TRT_PATH", "assets/model/HarnessOffDet_v0.0.0.engine"
)

LIMITED_NUM_OF_HARNESS_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_HARNESS_PER_CAMERA", 16))
OD_CONFIDENCE_THRESHOLD = 0.3
OD_NMS_THRESHOLD = 0.35


OD_TIME_INTERVAL_SECOND = 0.3
# 5초 동안 알람 유지 -> 현재 25개의 frame 관측
HARNESS_QUEUE_SIZE = 5
# 5초 측정동안 1초이상 이벤트 발생시 알람발생 -> 현재 5개의 frame이상시 알람
HARNESS_QUEUE_THRESHOLD = 1
HARNESS_CV_CATEGORY = ["harnessoff_cv", "하네스미착용_cv"]
OD_INPUT_SIZE = (1024, 1024)
