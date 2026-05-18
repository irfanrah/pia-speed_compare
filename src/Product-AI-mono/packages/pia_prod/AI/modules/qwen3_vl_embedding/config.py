import os

TEMPORAL_SIZE = 1
IMG_SIZE = (768,768) # (h,w)
DEVICE = "cuda"
QWEN3VLE_ID = "Qwen3-VL-Embedding-2B-FP8"

# Model paths
QWEN3VLE_MODEL_HF_PATH = os.getenv(
    "MODEL_QWEN3VLE_MODEL_HF_PATH", "assets/model/Qwen3-VL-Embedding-2B-FP8"
)

# Text Feature JSON
QWEN3VLE_TEXT_FEATURES_PATH = os.getenv(
    "QWEN3VLE_TEXT_FEATURES_PATH",
    "assets/model/Qwen3-VL-Embedding-2B-FP8/VLE_FP8_text_features.json",
)

# Event category keywords for each prediction type
FIRE_CATEGORY = ["fire_vle_ret", "화재_vle_ret"]
FALLDOWN_CATEGORY = ["falldown_vle_ret", "쓰러짐_vle_ret"]
VIOLENCE_CATEGORY = ["violence_vle_ret", "폭력_vle_ret"]
SMOKE_CATEGORY = ["smoke_vle_ret", "연기_vle_ret"]
ALL_CATEGORIES = FIRE_CATEGORY + FALLDOWN_CATEGORY + VIOLENCE_CATEGORY + SMOKE_CATEGORY

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "violence": VIOLENCE_CATEGORY,
    "smoke": SMOKE_CATEGORY,
}

QWEN3VLE_QUEUE_SIZE = int(os.getenv("QWEN3VLE_QUEUE_SIZE", 5))
QWEN3VLE_ALARM_DURATION_THRESHOLD = int(os.getenv("QWEN3VLE_ALARM_DURATION_THRESHOLD", 3))