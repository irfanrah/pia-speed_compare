import os

DEVICE = "cuda"
PERSON_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_ONNX_PATH", "assets/model/PersonDet_v3.7.0.onnx"
)
PERSON_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_TRT_PATH", "assets/model/PersonDet_v3.7.0.engine"
)

OD_CONFIDENCE_THRESHOLD = 0.5
OD_NMS_THRESHOLD = 0.35

OD_INPUT_SIZE = [640, 640]
OD_TARGET_CLASSES = [0]
OD_TIME_INTERVAL_SECOND = 0.3

DEFAULT_FPS = 15
LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", "16"))
LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", 16))
LIMITED_NUM_OF_TRACKED_OBJECTS = int(os.getenv("LIMITED_NUM_OF_TRACKED_OBJECTS", 32))

LOITERING_INTERVAL_SECOND = float(os.getenv("LOITERING_CV_TIME_INTERVAL", 0.3))
LOITERING_THRESHOLD_SECOND = float(os.getenv("LOITERING_THRESHOLD_SECOND", 10.0))

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


IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")


LOITERING_CV_CATEGORY = ["배회_cv", "loitering_cv"]
