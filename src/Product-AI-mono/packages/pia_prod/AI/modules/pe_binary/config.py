import os
import torch

IMG_SIZE = (336, 336)  # (height, width)
INPUT_SIZE = (3, *IMG_SIZE)
DEVICE = "cuda"
PE_BINARY_PYTORCH_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_PYTORCH_PATH", "assets/model/PE-Core-L14-336.pt"
)
PE_BINARY_ONNX_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_ONNX_PATH", "assets/model/PE-Core-L14-336.onnx"
)
PE_BINARY_TRT_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TRT_PATH", "assets/model/PE-Core-L14-336.engine"
)
PE_BINARY_TXT_FEATURE_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TXT_FEATURE_PATH", "assets/model/text_features.json"
)

INDEX_MAPPING = {
    0: "normal",
    1: "falldown",
    2: "fire",
    3: "smoke",
    4: "smoking",
    5: "esfalldown",
    6: "elvfalldown",
}

FIRE_CATEGORY = ["fire_ret", "화재_ret"]
FALLDOWN_CATEGORY = ["falldown_ret", "쓰러짐_ret"]
SMOKE_CATEGORY = ["smoke_ret", "연기_ret"]
SMOKING_CATEGORY = ["smoking_ret", "흡연_ret"]
ESFALLDOWN_CATEGORY = ["esfalldown_ret", "에스컬레이터쓰러짐_ret"]
ELVFALLDOWN_CATEGORY = ["elvfalldown_ret", "엘리베이터쓰러짐_ret"]
ALL_CATEGORIES = (
    FIRE_CATEGORY
    + FALLDOWN_CATEGORY
    + SMOKE_CATEGORY
    + SMOKING_CATEGORY
    + ESFALLDOWN_CATEGORY
    + ELVFALLDOWN_CATEGORY
)

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "smoke": SMOKE_CATEGORY,
    "smoking": SMOKING_CATEGORY,
    "esfalldown": ESFALLDOWN_CATEGORY,
    "elvfalldown": ELVFALLDOWN_CATEGORY,
}

TEMPORAL_SIZE = 1
INFERENCE_SEQUENCE_SIZE = 1
QUEUE_SIZE = int(os.environ.get("PE_QUEUE_SIZE", 5))
ALARM_DURATION_THRESHOLD = int(os.environ.get("PE_ALARM_DURATION_THRESHOLD", 3))

TOP_CANDIDATE = int(os.environ.get("PE_TOP_CANDIDATE", 13))
TOP_K = 4  # Not used

IMAGE_DTYPE = torch.float16
