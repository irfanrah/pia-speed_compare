import os

DEVICE = "cuda"
PERSON_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_ONNX_PATH", "assets/model/PersonDet_v3.7.0.onnx"
)
PERSON_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_PERSON_DETECTION_TRT_PATH", "assets/model/PersonDet_v3.7.0.engine"
)

WEAPON_DETECTION_MODEL_ONNX_PATH = os.getenv(
    "MODEL_WEAPON_DETECTION_ONNX_PATH", "assets/model/WeaponDet_v1.3.0.onnx"
)
WEAPON_DETECTION_MODEL_TRT_PATH = os.getenv(
    "MODEL_WEAPON_DETECTION_TRT_PATH", "assets/model/WeaponDet_v1.3.0.engine"
)

LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", "16"))
LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", 8))
OD_CONFIDENCE_THRESHOLD = 0.5
OD_NMS_THRESHOLD = 0.35
OD_INPUT_SIZE = [640, 640]
OD_TARGET_CLASSES = [0]
OD_TIME_INTERVAL_SECOND = 0.3
DEFAULT_FPS = 15

# Weapon detection configurations
WEAPON_DETECTION_QUEUE_SIZE = int(os.getenv("WEAPON_DETECTION_QUEUE_SIZE", 6))
WEAPON_DETECTION_QUEUE_THRESHOLD = int(os.getenv("WEAPON_DETECTION_QUEUE_THRESHOLD", 3))

# Weapon Detection configurations
WEAPON_DETECTION_CONFIDENCE_THRESHOLD = 0.35
TARGET_CATEGORY_INDEX = [1, 2, 3]

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")
WEAPON_CV_CATEGORY = ["weapon_cv", "무기소지_cv"]

# Detection Margin configurations
TOP_MARGIN_RATIO = 0.25
LEFT_MARGIN_RATIO = 0.5
RIGHT_MARGIN_RATIO = 0.5
BOTTOM_MARGIN_RATIO = 0.0

# Category Keys
CATEGORY_DICT = {0: "person", 1: "gun", 2: "knife", 3: "bat", 4: "phone"}

COLOR_DICT = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
    "pink": (203, 192, 255),
    "gray": (128, 128, 128),
}

CATEGORY_COLOR_DICT = {
    "person": COLOR_DICT["blue"],
    "gun": COLOR_DICT["black"],
    "knife": COLOR_DICT["red"],
    "bat": COLOR_DICT["yellow"],
    "phone": COLOR_DICT["purple"],
}
