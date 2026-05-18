import os

DEVICE = "cuda"
PERSON_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_ONNX_PATH", "assets/model/PersonDet_v3.7.0.onnx"
)
PERSON_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_TRT_PATH", "assets/model/PersonDet_v3.7.0.engine"
)
MODEL_HARNESS_CLS_ONNX_PATH = os.getenv(
    "MODEL_HARNESS_CLS_ONNX_PATH", "assets/model/BinaryHarnessCls_v0.2.1.onnx"
)
MODEL_HARNESS_CLS_TRT_PATH = os.getenv(
    "MODEL_HARNESS_CLS_TRT_PATH", "assets/model/BinaryHarnessCls_v0.2.1.engine"
)

OD_CONFIDENCE_THRESHOLD = 0.5
OD_NMS_THRESHOLD = 0.35

OD_INPUT_SIZE = [640, 640]
OD_TARGET_CLASSES = [0]
OD_TIME_INTERVAL_SECOND = 0.3

DEFAULT_FPS = 15
LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", "16"))
LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", "16"))

# Harness configurations
HARNESS_WITH_CLS_QUEUE_SIZE = int(os.getenv("HARNESS_WITH_CLS_QUEUE_SIZE", "12"))
HARNESS_WITH_CLS_QUEUE_THRESHOLD = int(os.getenv("HARNESS_WITH_CLS_QUEUE_THRESHOLD", "3"))

# Classification configurations
CLS_CONFIDENCE_THRESHOLD = 0.5
TARGET_CATEGORY_INDEX = (
    1  # 0: harness 1: no_harness
)
CLS_INPUT_SIZE = [224, 224]

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")


HARNESS_WITH_CLS_CV_CATEGORY = ["harnessoff_cv", "하네스미착용_cv"]
