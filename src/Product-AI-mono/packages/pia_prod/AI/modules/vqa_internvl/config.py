import os

# ── 모델 기본 경로 ──────────────────────────────────────────────────────────────
HF_REPO_ID = os.getenv("VQA_HF_REPO_ID", "OpenGVLab/InternVL3-2B")
ASSETS_MODEL_DIR = os.getenv("VQA_ASSETS_MODEL_DIR", "assets/model")
MODEL_DIR = os.getenv("MODEL_INTERNVL3_PATH", os.path.join(ASSETS_MODEL_DIR, HF_REPO_ID))

# VQAConfig.model_path 에 전달되는 값
INTERNVL3_MODEL_HF_PATH = MODEL_DIR

# ── VIT 엔진 경로 ───────────────────────────────────────────────────────────────
VIT_ONNX_FILE = os.getenv(
    "VQA_VIT_ONNX_FILE", os.path.join(MODEL_DIR, "vision_encoder_bfp16.onnx")
)
VIT_TRT_FILE = os.getenv(
    "VQA_VIT_TRT_FILE", os.path.join(MODEL_DIR, "vision_encoder_bfp16.trt")
)
VIT_IMAGE_PATH = os.getenv(
    "VQA_VIT_IMAGE_PATH", os.path.join(ASSETS_MODEL_DIR, "..", "images", "frame_0.jpg")
)

# ── VIT 엔진 빌드 설정 ──────────────────────────────────────────────────────────
VIT_DTYPE = os.getenv("VQA_VIT_DTYPE", "bfloat16")
VIT_MIN_BS = int(os.getenv("VQA_VIT_MIN_BS", 1))
VIT_OPT_BS = int(os.getenv("VQA_VIT_OPT_BS", 13))
VIT_MAX_BS = int(os.getenv("VQA_VIT_MAX_BS", 32))

# ── LLM 체크포인트 / 엔진 경로 ─────────────────────────────────────────────────
TLLM_CHECKPOINT_DIR = os.getenv(
    "VQA_TLLM_CHECKPOINT_DIR", os.path.join(MODEL_DIR, "tllm_checkpoint")
)
TRT_ENGINE_DIR = os.getenv(
    "VQA_TRT_ENGINE_DIR", os.path.join(MODEL_DIR, "trt_engines")
)

# ── LLM 엔진 빌드 설정 ─────────────────────────────────────────────────────────
LLM_DTYPE = os.getenv("VQA_LLM_DTYPE", "bfloat16")
LOAD_MODEL_ON_CPU = os.getenv("VQA_LOAD_MODEL_ON_CPU", "true").lower() == "true"
DISABLE_ROPE_SCALING = os.getenv("VQA_DISABLE_ROPE_SCALING", "true").lower() == "true"

LLM_MAX_BATCH_SIZE = int(os.getenv("VQA_LLM_MAX_BATCH_SIZE", 2))
LLM_MAX_INPUT_LEN = int(os.getenv("VQA_LLM_MAX_INPUT_LEN", 7312))
LLM_MAX_SEQ_LEN = int(os.getenv("VQA_LLM_MAX_SEQ_LEN", 7326))
LLM_MAX_NUM_TOKENS = int(os.getenv("VQA_LLM_MAX_NUM_TOKENS", 20000))
LLM_GEMM_PLUGIN = os.getenv("VQA_LLM_GEMM_PLUGIN", "bfloat16")
LLM_MAX_MULTIMODAL_LEN = int(os.getenv("VQA_LLM_MAX_MULTIMODAL_LEN", 23040))

# ── LLM 런타임 동작 ─────────────────────────────────────────────────────────────
LLM_FORCE_SINGLE_BATCH = os.getenv("VQA_LLM_FORCE_SINGLE_BATCH", "false").lower() == "true"
LLM_BATCH_USE_EXECUTOR = os.getenv("VQA_LLM_BATCH_USE_EXECUTOR", "true").lower() == "true"
LLM_BATCH_USE_ASYNC = os.getenv("VQA_LLM_BATCH_USE_ASYNC", "true").lower() == "true"
LLM_USE_INPUT_TOKEN_EXTRA_IDS = (
    os.getenv("VQA_LLM_USE_INPUT_TOKEN_EXTRA_IDS", "false").lower() == "true"
)

# ── 추론 공통 설정 ──────────────────────────────────────────────────────────────
N_GPUS = int(os.getenv("VQA_N_GPUS", 1))
FRAME_PER_TILE_MAX_NUM = int(os.getenv("VQA_FRAME_PER_TILE_MAX_NUM", 16))
DEVICE = "cuda"
MODEL_INF_DATA_TYPE = os.getenv("VQA_MODEL_INF_DATA_TYPE", "bfloat16")
MAX_NEW_TOKEN = int(os.getenv("VQA_MAX_NEW_TOKEN", 13))

# ── 카테고리 ────────────────────────────────────────────────────────────────────
FIRE_CATEGORY = ["fire_vqa", "화재_vqa"]
FALLDOWN_CATEGORY = ["falldown_vqa", "쓰러짐_vqa"]
MODEL_OUTPUTS = ["falldown", "fire"]
SUPPORT_CATEGORIES = FALLDOWN_CATEGORY

# ── 이벤트 판정 설정 ────────────────────────────────────────────────────────────
QUEUE_SIZE = int(os.environ.get("VQA_FALLDOWN_QUEUE_SIZE", 1))
ALARM_DURATION_THRESHOLD = int(os.environ.get("VQA_FALLDOWN_ALARM_DURATION_THRESHOLD", 1))
