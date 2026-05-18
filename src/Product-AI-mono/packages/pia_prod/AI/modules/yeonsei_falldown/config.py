import os

PERSON_KEYPOINT_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PERSON_KEYPOINT_ONNX_PATH", "assets/model/PersonKP_v1.0.0.onnx"
)
PERSON_KEYPOINT_MODEL_TRT_PATH = os.getenv(
    "MODEL_PERSON_KEYPOINT_TRT_PATH", "assets/model/PersonKP_v1.0.0.engine"
)
KP_INPUT_SIZE = (640, 640)
DEVICE = "cuda"
LETTER_BOX_COLOR = (114, 114, 114)
BBOX_CUTTING_MARGIN_RATIO = 0.03  # float(os.getenv("FALL_BBOX_CUTTING_MARGIN_RATIO")) # 0.03
BBOX_KEEP_HEAD_FOOT_RATIO = 0.21  # float(os.getenv("FALL_BBOX_KEEP_HEAD_FOOT_RATIO")) # 0.21
ANGLE_THRESHOLD = 30  # int(os.getenv("FALL_ANGLE_THRESHOLD")) # 30
FALLDOWN_CV_CATEGORY = ["falldown_cv", "쓰러짐_cv"]

# Alarm Event config
FALL_QUEUE_SIZE = int(
    os.getenv("FALLDOWN_CV_QUEUE_SIZE", 10)
)  # Default queue size for fall detection
ALARM_DURATION = int(
    os.getenv("FALLDOWN_CV_DURATION_THRESHOLD", 3)
)  # Default threshold for alarm duration
