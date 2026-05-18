import os
import torch

IMG_SIZE = (336, 336)  # (height, width)
INPUT_SIZE = (3, *IMG_SIZE)
DEVICE = "cuda"
PERCEPTION_ENCODER_PYTORCH_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_PYTORCH_PATH", "assets/model/PE-Core-L14-336.pt"
)
PERCEPTION_ENCODER_ONNX_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_ONNX_PATH", "assets/model/PE-Core-L14-336.onnx"
)
PERCEPTION_ENCODER_TRT_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TRT_PATH", "assets/model/PE-Core-L14-336.engine"
)
PERCEPTION_ENCODER_TXT_FEATURE_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TXT_FEATURE_PATH", "assets/model/text_features.json"
)

INDEX_MAPPING = {0: "normal", 1: "falldown", 2: "fire", 3: "smoke", 4: "smoking"}

FIRE_CATEGORY = ["fire_ret", "화재_ret"]
FALLDOWN_CATEGORY = ["falldown_ret", "쓰러짐_ret"]
SMOKE_CATEGORY = ["smoke_ret", "연기_ret"]
SMOKING_CATEGORY = ["smoking_ret", "흡연_ret"]
ALL_CATEGORIES = FIRE_CATEGORY + FALLDOWN_CATEGORY + SMOKE_CATEGORY + SMOKING_CATEGORY

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "smoke": SMOKE_CATEGORY,
    "smoking": SMOKING_CATEGORY,
}

TEMPORAL_SIZE = 1
INFERENCE_SEQUENCE_SIZE = 1
QUEUE_SIZE = int(os.environ.get("PE_QUEUE_SIZE", 10))
ALARM_DURATION_THRESHOLD = int(os.environ.get("PE_ALARM_DURATION_THRESHOLD", 5))

TOP_CANDIDATE = int(os.environ.get("PE_TOP_CANDIDATE", 13))
TOP_K = 4  # Not used

INTERNVL3_VQA_CATEGORY = ["화재_vqa", "fire_vqa", "쓰러짐_vqa", "falldown_vqa"]
IMAGE_DTYPE = torch.float16
