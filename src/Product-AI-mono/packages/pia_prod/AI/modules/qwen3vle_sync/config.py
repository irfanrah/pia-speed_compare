import os

TEMPORAL_SIZE = 1
IMG_SIZE = (768, 768)  # (h, w)
DEVICE = "cuda"

# --- HF / asset paths --------------------------------------------------------
QWEN3VLE_SYNC_ID = "Qwen3-VL-Embedding-2B-FP8"

QWEN3VLE_SYNC_MODEL_PATH = os.getenv(
    "QWEN3VLE_SYNC_MODEL_PATH",
    "assets/model/Qwen3-VL-Embedding-2B-FP8",
)

QWEN3VLE_SYNC_TEXT_FEATURES_PATH = os.getenv(
    "QWEN3VLE_SYNC_TEXT_FEATURES_PATH",
    "assets/model/Qwen3-VL-Embedding-2B-FP8/VLE_FP8_text_features.json",
)

# --- vLLM engine tuning ------------------------------------------------------
QWEN3VLE_SYNC_DTYPE = os.getenv("QWEN3VLE_SYNC_DTYPE", "auto")
QWEN3VLE_SYNC_GPU_MEMORY_UTILIZATION = float(
    os.getenv("QWEN3VLE_SYNC_GPU_MEMORY_UTILIZATION", "0.3")
)
QWEN3VLE_SYNC_MAX_MODEL_LEN = int(os.getenv("QWEN3VLE_SYNC_MAX_MODEL_LEN", "8192"))
QWEN3VLE_SYNC_ENFORCE_EAGER = (
    os.getenv("QWEN3VLE_SYNC_ENFORCE_EAGER", "false").lower() == "true"
)

# Qwen3-VL video processor uses temporal_factor=2; T=1 inputs are padded
# inside `_build_video_input` by repeating the last frame. Identical to
# the embedding_server's behavior so the embeddings are byte-comparable.
TEMPORAL_FACTOR = 2

DEFAULT_INSTRUCTION = os.getenv(
    "QWEN3VLE_SYNC_DEFAULT_INSTRUCTION", "Represent the user's input."
)

# --- Event category keywords (must match qwen3vle for cross-module reuse) ---
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

# --- Event-manager tuning (same defaults as qwen3vle) ------------------------
QWEN3VLE_SYNC_QUEUE_SIZE = int(os.getenv("QWEN3VLE_SYNC_QUEUE_SIZE", 5))
QWEN3VLE_SYNC_ALARM_DURATION_THRESHOLD = int(
    os.getenv("QWEN3VLE_SYNC_ALARM_DURATION_THRESHOLD", 3)
)
