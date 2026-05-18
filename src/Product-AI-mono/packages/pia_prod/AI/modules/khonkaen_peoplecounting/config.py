import os

PEOPLE_THRESHOLD = 100
# ALARM_THRESHOLD = 3
# CROWD_PEOPLE_QUEUE_SIZE = 5
CROWD_PEOPLE_ONNX_MODEL_PATH = os.getenv(
    "MODEL_CROWD_PEOPLE_ONNX_PATH", "assets/model/CLIP_EBC_nwpu_rmse_onnx.onnx"
)
DEVICE = "cuda"
PEOPLECOUNTING_CV_CATEGORY = ["peoplecounting_cv", "피플카운팅_cv"]
DEFAULT_PEOPLECOUNTING_INTERVAL = 5  # 5seconds interval for checking counting result
