import os

DEVICE = "cuda"
MODEL_VEHICLE_REVERSE_ONNX_PATH = os.getenv(
    "MODEL_VEHICLE_REVERSE_ONNX_PATH", "assets/model/VehicleDet_v0.1.1.onnx"
)
MODEL_VEHICLE_REVERSE_TRT_PATH = os.getenv(
    "MODEL_VEHICLE_REVERSE_TRT_PATH", "assets/model/VehicleDet_v0.1.1.engine"
)
REID_MODEL_PATH = os.getenv(
    "MODEL_VEHICLE_REVERSE_REID_PT_PATH", "assets/model/vehicle_reid_deploy.pt"
)
TRACKER_CONFIG_PATH = os.getenv("VEHICLE_REVERSE_TRACKER_CONFIG_PATH", "")

OD_CONFIDENCE_THRESHOLD = float(os.getenv("VEHICLE_REVERSE_CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("VEHICLE_REVERSE_IOU_THRESHOLD", "0.6"))
ROI_MODE = os.getenv("VEHICLE_REVERSE_ROI_MODE", "center")
DEBUG_MODE = True
DETECTION_INPUT_SIZE = (640, 640)

TRACK_CLASSES = [0, 1, 2]
TRACKER_CONFIG = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.5,
    "max_age": 90,
    "max_iou_distance": 0.2,
    "gmc_method": "none",
    "proximity_thresh": 0.01,
    "appearance_thresh": 0.6,
    "fuse_score": True,
    "with_reid": True,
}

# LANE_POLYGONS = [
#     [
#         [30.104, 673.932],
#         [461.732, 584.987],
#         [1105.28, 459.688],
#         [1404.544, 398.991],
#         [1507.22, 382.546],
#         [1593.6, 385.273],
#         [1547.298, 499.629],
#         [1471.016, 713.62],
#         [1355.586, 1077.689],
#         [32.728, 1079.0],
#     ]
# ]  # [[20, 1074], [1357, 1075], [1608, 374], [1541, 368], [6, 666]]

# LANE_DIRECTIONS = [
#     [-0.8963189244454537, 0.4434099521672299]
# ]  # [[1584, 372], [200, 1077]] 으로 방향 벡터를 구할 수 있음

WRONGWAY_PARAMS = {
    "fps": 8,
    "vel_win": 2,
    "speed_min": 20.0,
    "cos_on": -0.55,
    "cos_off": 0.1,
    "consec_on": 3,
    "grace_off": 3,
    "avg_win": 4,
    "lane_stable": 4,
    "max_idle": 20,
    "homography": None,
}

EVENT_QUEUE_SIZE = int(os.getenv("VEHICLE_REVERSE_QUEUE_SIZE", "16"))
EVENT_QUEUE_THRESHOLD = int(os.getenv("VEHICLE_REVERSE_QUEUE_THRESHOLD", "7"))

CATEGORY_NAME = "vehiclereverse_cv"
VEHICLE_REVERSE_CV_CATEGORY = ["vehiclereverse_cv", "차량역주행_cv"]
DEFAULT_FPS = 0.125  # 8fps

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")