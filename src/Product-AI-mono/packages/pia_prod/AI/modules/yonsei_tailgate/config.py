import os

DEVICE = "cuda"
PERSON_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_ONNX_PATH", "assets/model/PersonDet_v3.7.0.onnx"
)
PERSON_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_TRT_PATH", "assets/model/PersonDet_v3.7.0.engine"
)

MODEL_WALLJUMP_CLS_ONNX_PATH = os.getenv(
    "MODEL_WALLJUMP_CLS_ONNX_PATH", "assets/model/PersonWalljumpCls_v2.7.0.onnx"
)
MODEL_WALLJUMP_CLS_TRT_PATH = os.getenv(
    "MODEL_WALLJUMP_CLS_TRT_PATH", "assets/model/PersonWalljumpCls_v2.7.0.engine"
)


LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", 16))
OD_NMS_THRESHOLD = 0.35

OD_INPUT_SIZE = [640, 640]
OD_TARGET_CLASSES = [0]
OD_TIME_INTERVAL_SECOND = 0.3

DEFAULT_FPS = 15

# Tailgate configurations
VARIANCE_THRESHOLD_FOR_TAILGATE = float(
    os.getenv("VARIANCE_THRESHOLD_FOR_TAILGATE", 1.6)
)  # 기본값 1.6, 낮을수록 오탐증가 미탐감소, 높을수록 오탐감소 미탐증가 -> 거리/속도 관련
TAIL_QUEUE_WINDOW_SIZE = 9
TAILGATE_CONFIDENCE_THRESHOLD = float(
    os.getenv("TAILGATE_CONFIDENCE_THRESHOLD", 0.5)
)  # 기본값 0.5, 낮을수록 오탐증가 미탐감소, 높을수록 오탐감소 미탐증가 -> 가방관련

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")
TAILGATE_CV_CATEGORY = ["tailgate_cv", "따라들어가기_cv"]
