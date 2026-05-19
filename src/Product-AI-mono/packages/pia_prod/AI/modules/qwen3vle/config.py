import os

TEMPORAL_SIZE = 1
IMG_SIZE = (768,768) # (h,w)
DEVICE = "cuda"
QWEN3VLE_ID = "Qwen3-VL-Embedding-2B-FP8"

# vLLM Embedding Server endpoint. Service.py POSTs JPEG-encoded frame batches
# to /v1/embeddings/batch and receives pooled video embeddings back. The
# weights live inside the embedding_server/ Docker container, not here.
QWEN3VLE_EMBEDDING_SERVER_URL = os.getenv(
    "QWEN3VLE_EMBEDDING_SERVER_URL", "http://localhost:8210"
)
QWEN3VLE_EMBEDDING_TIMEOUT = float(os.getenv("QWEN3VLE_EMBEDDING_TIMEOUT", "30"))
QWEN3VLE_EMBEDDING_JPEG_QUALITY = int(os.getenv("QWEN3VLE_EMBEDDING_JPEG_QUALITY", "95"))

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