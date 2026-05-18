import os

DEVICE = "cuda"
MODEL_BIKER_DETECTION_ONNX_PATH = os.getenv(
    "MODEL_BIKER_DETECTION_ONNX_PATH", "assets/model/PersonBikerDet_v3.0.0.onnx"
)
MODEL_BIKER_DETECTION_TRT_PATH = os.getenv(
    "MODEL_BIKER_DETECTION_TRT_PATH", "assets/model/PersonBikerDet_v3.0.0.engine"
)

MODEL_HELMET_CLS_ONNX_PATH = os.getenv(
    "MODEL_HELMET_CLS_ONNX_PATH", "assets/model/PersonHelmetCls_v1.4.0.onnx"
)
MODEL_HELMET_CLS_TRT_PATH = os.getenv(
    "MODEL_HELMET_CLS_TRT_PATH", "assets/model/PersonHelmetCls_v1.4.0.engine"
)

OD_CONFIDENCE_THRESHOLD = 0.5
OD_NMS_THRESHOLD = 0.35

OD_INPUT_SIZE = [640, 640]
OD_TARGET_CLASSES = [0]
OD_TIME_INTERVAL_SECOND = 0.3

DEFAULT_FPS = 15
LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", "16"))
LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", 16))

# Helmet configurations
HELMET_QUEUE_SIZE = int(os.getenv("HELMET_QUEUE_SIZE", 12))
HELMET_QUEUE_THRESHOLD = int(os.getenv("HELMET_QUEUE_THRESHOLD", 3))

# Classification configurations
CLS_CONFIDENCE_THRESHOLD = 0.5
TARGET_CATEGORY_INDEX = (
    1  # 0: bikerwithhelmet, 1: bikerwithouthelmet, 2: pedestrian, 3: pedestrianwithhelmet
)
CLS_INPUT_SIZE = [224, 224]

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")


HELMET_CV_CATEGORY = ["헬멧_cv", "helmet_cv"]
