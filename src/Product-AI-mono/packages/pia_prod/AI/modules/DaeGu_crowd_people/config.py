import os

PEOPLE_THRESHOLD = 50
ALARM_THRESHOLD = 3
CROWD_PEOPLE_QUEUE_SIZE = 5
CROWD_PEOPLE_ONNX_MODEL_PATH = os.getenv(
    "MODEL_CROWD_PEOPLE_ONNX_PATH", "assets/model/CLIP_EBC_nwpu_rmse_onnx.onnx"
)
DEVICE = "cuda"
CROWDED_CV_CATEGORY = ["crowded_cv", "군중밀집_cv"]
