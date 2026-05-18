"""
PE-VLE (PE TRT → Qwen3VLE Validation Server) module config.

Mirrors pe_vqa_2stage's config layout: Stage 1 = PE TensorRT, Stage 2 =
fire-and-forget HTTP POST to a validation server. The validation server
(`validation_server/`) holds all Stage-2 policy: in-process vLLM embed,
anchor classification, and the publish path.
"""

import os
from typing import Dict, List

# =====================================================================
# 1. Stage 1 (PE) inference constants
# =====================================================================
IMG_SIZE = (336, 336)  # (height, width)
INPUT_SIZE = (3, *IMG_SIZE)
DEVICE = os.getenv("PE_VLE_DEVICE", "cuda")

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

INDEX_MAPPING: Dict[int, str] = {
    0: "normal", 1: "falldown", 2: "fire", 3: "smoke"
}

TEMPORAL_SIZE = 1
INFERENCE_SEQUENCE_SIZE = 1
QUEUE_SIZE = int(os.environ.get("PE_VLE_QUEUE_SIZE", 10))
ALARM_DURATION_THRESHOLD = int(os.environ.get("PE_VLE_ALARM_DURATION_THRESHOLD", 5))

TOP_CANDIDATE = int(os.environ.get("PE_VLE_TOP_CANDIDATE", 13))
TOP_K = 4

IMAGE_DTYPE = "float16"


# =====================================================================
# 2. Category Event Mapping (PE-VLE labels)
# =====================================================================
FIRE_CATEGORY: List[str]     = ["fire_pe_vle_ret", "화재_pe_vle_ret"]
FALLDOWN_CATEGORY: List[str] = ["falldown_pe_vle_ret", "쓰러짐_pe_vle_ret"]
SMOKE_CATEGORY: List[str]    = ["smoke_pe_vle_ret", "연기_pe_vle_ret"]
ALL_CATEGORIES: List[str] = FIRE_CATEGORY + FALLDOWN_CATEGORY + SMOKE_CATEGORY

CATEGORY_EVENT_MAP: Dict[str, List[str]] = {
    "fire": FIRE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "smoke": SMOKE_CATEGORY,
}


# =====================================================================
# 3. Stage 2 (Validation Server) backend — pe_vqa_2stage shape
# =====================================================================
# Master toggle. False → behave like a plain PE module: every alarm publishes
# directly via match_outputs. True → categories in TWO_STEP_CATEGORIES are
# delegated fire-and-forget to the validation server.
PE_VLE_VALIDATION_ENABLED = os.getenv(
    "PE_VLE_VALIDATION_ENABLED", "true"
).lower() == "true"

VALIDATION_HOST = os.getenv("PE_VLE_VALIDATION_HOST", "localhost")
VALIDATION_PORT = int(os.getenv("PE_VLE_VALIDATION_PORT", "8200"))
VALIDATION_SERVER_URL = f"http://{VALIDATION_HOST}:{VALIDATION_PORT}"
VALIDATION_SERVER_ENDPOINT = os.getenv(
    "PE_VLE_VALIDATION_SERVER_ENDPOINT", "/api/v1/validate"
)
VALIDATION_SERVER_TIMEOUT = float(os.getenv("PE_VLE_VALIDATION_SERVER_TIMEOUT", "30.0"))

# Categories subject to Stage-2 verification. Default = fire + smoke + falldown
# (all PE-VLE categories that the validation server holds anchor buckets for).
_TWO_STEP_CATEGORIES_RAW = os.getenv(
    "TWO_STEP_CATEGORIES",
    ",".join(FIRE_CATEGORY + SMOKE_CATEGORY + FALLDOWN_CATEGORY),
)
TWO_STEP_CATEGORIES = set(c.strip() for c in _TWO_STEP_CATEGORIES_RAW.split(",") if c.strip())
