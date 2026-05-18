import os

DEVICE = "cuda"
MODEL_PATIENT_DETECTION_ONNX_PATH = os.getenv(
    "MODEL_PATIENT_DETECTION_ONNX_PATH", "assets/model/PatientDet_v0.1.0.onnx"
)
MODEL_PATIENT_DETECTION_TRT_PATH = os.getenv(
    "MODEL_PATIENT_DETECTION_TRT_PATH", "assets/model/PatientDet_v0.1.0.engine"
)

MODEL_CLOTHCOLOR_CLS_ONNX_PATH = os.getenv(
    "MODEL_CLOTHCOLOR_CLS_ONNX_PATH", "assets/model/ClothcolorCls_v0.1.0.onnx"
)
MODEL_CLOTHCOLOR_CLS_TRT_PATH = os.getenv(
    "MODEL_CLOTHCOLOR_CLS_TRT_PATH", "assets/model/ClothcolorCls_v0.1.0.engine"
)

OD_CONFIDENCE_THRESHOLD = 0.5
OD_NMS_THRESHOLD = 0.35

OD_INPUT_SIZE = [640, 640]
OD_TARGET_CLASSES = [0]
OD_TIME_INTERVAL_SECOND = 0.3

DEFAULT_FPS = 15
LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", "16"))
LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", 16))
LIMITED_NUM_OF_TRACKED_OBJECTS = int(os.getenv("LIMITED_NUM_OF_TRACKED_OBJECTS", 16))

# Helmet configurations
PATIENT_QUEUE_SIZE = int(os.getenv("PATIENT_QUEUE_SIZE", 5))
PATIENT_QUEUE_THRESHOLD = int(os.getenv("PATIENT_QUEUE_THRESHOLD", 3))

# Classification configurations
CLS_CONFIDENCE_THRESHOLD = 0.3
CLASSIFY_DICT = {
    0: "beige",
    1: "black",
    2: "blue",
    3: "blue_green",
    4: "brown",
    5: "burgundy",
    6: "dark_gray",
    7: "gray",
    8: "green",
    9: "khaki",
    10: "light_brown",
    11: "navy",
    12: "orange",
    13: "pink",
    14: "purple",
    15: "red",
    16: "red_brown",
    17: "sky_blue",
    18: "white",
    19: "yellow",
    20: "yellow_green",
}

# TRACKER
TRACKER_DICT = {
    "det_thresh": 0.5,
    "max_age": int(os.environ.get("PATIENCE_TRACKER_MAX_AGE", 10)),
    "min_hits": int(os.environ.get("PATIENCE_TRACKER_MIN_HITS", 3)),
    "iou_threshold": 0.3,
    "delta_t": 3,
    "asso_func": "iou",
    "inertia": 0.2,
    "use_byte": bool(os.environ.get("PATIENCE_TRACKER_USE_BYTE", 1)),
}

TARGET_CATEGORY_INDEX = (
    12,
    13,
    15,
    17,  # 2: blue(normal), 12: orange, 13: pink, 15: red, 17: sky_blue
)
CLS_INPUT_SIZE = [224, 224]

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")


PATIENT_CV_CATEGORY = ["환자배회_cv", "patient_cv"]
