import os

TEMPORAL_SIZE = 1
IMG_SIZE = (768, 768)  # (h, w) — must match the fixed resolution baked into the ONNX export
DEVICE = "cuda"

# HuggingFace model directory (used for the Qwen3VLProcessor / tokenizer for export).
QWEN3VLE_TRT_PT_MODEL_PATH = os.getenv(
    "QWEN3VLE_TRT_PT_MODEL_PATH", "assets/model/Qwen3-VL-Embedding-2B"
)

# Directory containing the TRT engine files + rotary_params.npz.
# Must contain Vision.engine, Transformer.engine, rotary_params.npz.
QWEN3VLE_TRT_ID = "Qwen3-VL-Embedding-2B-onnx"
QWEN3VLE_TRT_ONNX_DIR_PATH = os.getenv(
    "QWEN3VLE_TRT_ONNX_DIR_PATH", "assets/model/Qwen3-VL-Embedding-2B-onnx"
)

# Text feature path
QWEN3VLE_TRT_TEXT_FEATURES_PATH = os.getenv(
    "QWEN3VLE_TRT_TEXT_FEATURES_PATH",
    "assets/model/Qwen3-VL-Embedding-2B-onnx/VLE_BF16_text_features.json",
)
# Event category keywords
FIRE_CATEGORY = ["fire_vle_ret", "화재_vle_ret"]
SMOKE_CATEGORY = ["smoke_vle_ret", "연기_vle_ret"]
ALL_CATEGORIES = FIRE_CATEGORY + SMOKE_CATEGORY

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "smoke": SMOKE_CATEGORY,
}

QWEN3VLE_QUEUE_SIZE = int(os.getenv("QWEN3VLE_TRT_QUEUE_SIZE", 5))
QWEN3VLE_ALARM_DURATION_THRESHOLD = int(os.getenv("QWEN3VLE_TRT_ALARM_DURATION_THRESHOLD", 3))
